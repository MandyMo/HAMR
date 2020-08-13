## HAMR
<p align="center">
 <img src="./images/mesh.png" width="800px">
</p>

This repo is the source code for [End-to-end Hand Mesh Recovery from a Monocular RGB Image](https://arxiv.org/abs/1902.09305) by *Xiong Zhang, Qiang Li, Hong Mo, Webnbo Zhang*, and *Wen Zheng*. HAMR targets at tackle the problem of reconstructing the full 3D mesh of a human hand from a single RGB image. In contrast to existing research on 2D or 3D hand pose estimation from RGB or/and depth image data, HAMR can provide a more expressive and useful mesh representation for monocular hand image understanding. In particular, the mesh representation is achieved by parameterizing a generic 3D hand model with shape and relative 3D joint angles.



### marks
We shall apologize to the reviewer of our paper when we submit it to ICCV2019.
The following are selected review comments from reviewer1,
The authors mistakenly insist that their Equation 3 is right. Different regressors are applied differently ([16] uses 2 different ones), but there is only 1 correct way for each. The source code of [16] (cited in the rebuttal at L032) verifies my comment, please see:
- https://github.com/akanazawa/hmr/blob/ffc0297872779031ec9b2ab87d7e9843b3cf8c90/src/tf_smpl/batch_smpl.py#L115 (MANO joint regressor)
- https://github.com/akanazawa/hmr/blob/ffc0297872779031ec9b2ab87d7e9843b3cf8c90/src/tf_smpl/batch_smpl.py#L152 (COCO joint regressor)
In reality Eq3 is a non-destructive heuristic, that looks plausible and training works, therefore I am *not* attacking for rejection based on this. This does not change in the paper now. But I strongly and kindly suggest to add a comment in your source-code that this is not the optimal/suggested way, otherwise people that copy/borrow your code or follow Eq3 might be following the non-correct way. Potentially you could remove Eq3 and simply cite the paper (good idea of other reviewer).

Only recently, we found that our Equation 3 is not correct.

### Citation
If you use this code for your research, please cite:
```
@article{zhang2019end,
  title={End-to-end Hand Mesh Recovery from a Monocular RGB Image},
  author={Zhang, Xiong and Li, Qiang and Mo, Hong and Zhang, Wenbo and Zheng, Wen},
  booktitle = {The International Conference on Computer Vision (ICCV)},
  year = {2019}
}
```
