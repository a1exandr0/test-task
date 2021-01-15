# test-task

## Extends Internal classes from https://github.com/clovaai/stargan-v2 repo in order to make it possible to send HTTP requests
In order to make everithing work you colud refer to helper.sh and requirements from initial repo.
Could be deployed on a machine with conda env
Based on Flask and original stargan-v2 repo

### In order to make api.py work path it to root folder of original repo and run without any parameters and install all requirements for CelebA-HQ dataset

1. Clone the origignal stargan-v2 repo linked above.
1. Install all required modules.
1. Pull pretrained on CelebA-HQ model using download.sh script.
1. Run api.py without any arguments from repo root folder.
1. Now you are good to send requests.

Send them in a form {"src": some_val: List, "ref": some_val: List, "ref_label": some_val: Int}.
where:
* "src":source image in a form of list
* "ref": reference image in a from of list
* "ref_label":label reference for style generation integer 0 for female or 1 for male reference
