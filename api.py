from flask import Flask, jsonify, request
from PIL import Image
from core.solver import Solver
from torch.backends import cudnn
import torch
import argparse
from core.utils import denormalize
from torchvision import transforms
import numpy as np
from core.wing import FaceAligner


app = Flask(__name__)   # Flask app __init__


def transform(inp, img_size):
    """
    Transforms given image into appropriate format for the model.
    Gets np.array and desired image size as inputs.
    """
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    t = t(inp)
    return torch.unsqueeze(t, 0)


def tensor_to_image(img):
    """
    Transforms torch.Tensor to PIL Image.
    Gets torch.Tensor as input.
    """
    t = transforms.ToPILImage()
    return t(img)


class ModifiedSolver(Solver):
    """
    Overridden Solver class from Git repo, makes it possible to get inference of np.array forms of input images.
    """
    def __init__(self, args):
        super(ModifiedSolver, self).__init__(args)
        self._load_checkpoint(self.args.resume_iter)

    def align_preprocessed(self, img):
        """
        Aligns face on image according to landmarks.
        Gets torch.Tensor as input.
        """
        aligner = FaceAligner(self.args.wing_path, self.args.lm_path, self.args.img_size)
        return aligner.align(img)

    @torch.no_grad()
    def paired_request(self, src_img, ref_img, ref_label):
        """
        Here is where all the inference process happens.
        Gets source and inference images as np.arrays. and label
        which is int(1) for male reference photo and int(0) for female.
        Returns synthesized image in PIL Image form.
        """
        nets = self.nets_ema
        src_img = transform(src_img, self.args.img_size).to(self.device)
        ref_img = transform(ref_img, self.args.img_size).to(self.device)

        src_img = self.align_preprocessed(src_img)
        ref_img = self.align_preprocessed(ref_img)

        masks = nets.fan.get_heatmap(src_img) if self.args.w_hpf > 0 else None
        s_ref = nets.style_encoder(ref_img, torch.from_numpy(np.array(ref_label)).to(self.device))
        x_fake = nets.generator(src_img, s_ref, masks=masks)
        x_fake = denormalize(x_fake)
        return tensor_to_image(x_fake.squeeze(0).cpu())


def main(args):
    cudnn.benchmark = True          #
    torch.manual_seed(args.seed)    # Setting up Solver class for making forward passes
    solver = ModifiedSolver(args)   #

    def get_inference(src_img, ref_img, ref_label):
        """
        Helper function for serving REST requests.
        """
        res = solver.paired_request(src_img, ref_img, [ref_label])
        return np.array(res)

    @app.route("/inference", methods=["POST"])
    def inference():
        """
        Main function for serving REST prediction requests.
        """
        if request.method == "POST":
            data = request.json                                 #
            src_img = np.array(data["src"]).astype(np.uint8)    # Parsing data
            ref_img = np.array(data["ref"]).astype(np.uint8)    #
            ref_label = int(data["ref_label"])                  #
            result = get_inference(src_img, ref_img, ref_label) # Calling helper function
            return jsonify({"result": result.tolist()})         # Returning results into json

    app.run()   # Starting app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=100000,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--mode', type=str, default="sample",
                        choices=['train', 'sample', 'eval', 'align'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val',
                        help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='expr/samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints/celeba_hq',
                        help='Directory for saving network checkpoints')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')

    # face alignment
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=50000)

    args = parser.parse_args()
    main(args)
