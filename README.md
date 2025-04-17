# Land Cover Classification from Remote Sensing Images Based on Multi-Scale Fully Convolutional Network


In this repository, we design two branches with convolutional layers in different kernel sizes in each layer of the encoder to capture multi-scale features. Besides, a channel attention block and a global pooling module are utilized to enhance channel consistency and global contextual consistency. Substantial experiments are conducted on both 2D RGB images datasets and 3D spatial-temporal datasets.

The detailed results can be seen in the [Land Cover Classification from Remote Sensing Images Based on Multi-Scale Fully Convolutional Network](https://www.tandfonline.com/doi/full/10.1080/10095020.2021.2017237).

The training and testing code can refer to [GeoSeg](https://github.com/rajesh0305/GeoSeg-main).

The related repositories include:
* [MACU-Net](https://github.com/lironui/MACU-Net)->A revised U-Net structure.
* [MAResU-Net](https://github.com/lironui/MAResU-Net)->Another type of attention mechanism with linear complexity.

If our code is helpful to you, please cite:

`Li, R., Zheng, S., Duan, C., Wang, L., & Zhang, C. (2021). Land Cover Classification from Remote Sensing Images Based on Multi-Scale Fully Convolutional Network. Geo-spatial Information Science.`

Acknowlegement:
------- 
Thanks to the providers of the following open-source datasets:

[WHDLD](https://sites.google.com/view/zhouwx/dataset?authuser=0#h.p_ebsAS1Bikmkd)

[GID](https://x-ytong.github.io/project/GID.html)

[2015&2017](http://gpcv.whu.edu.cn/data/3DFGC_pages.html)

Requirements：
------- 
```
numpy >= 1.16.5
PyTorch >= 1.3.1
sklearn >= 0.20.4
tqdm >= 4.46.1
imageio >= 2.8.0
```

Network:
------- 
![network](https://github.com/rajesh0305/MSFCN_Results/blob/main/network.png)  
Fig. 1.  The structure of the proposed Multi-Scale Fully Convolutional Network.

Result:
------- 
![Result1](https://github.com/rajesh0305/MSFCN_Results/blob/main/2D_zoom%20(1).png)  
Fig. 2. Visualization of results on the WHDLD and GID datasets.

Performence Metrics:
-------
![Performrnce Metrics](https://github.com/rajesh0305/MSFCN_Results/blob/main/whdld%20result.png)
Fig. 3. Performence Metrices of the proposed MSFCN on the WHDLD Dataset.

Computational Complexity Analysis:
-------
![Computational Complexity Analysis](https://github.com/rajesh0305/MSFCN_Results/blob/main/computational%20complexity%20analysis.png)
Fig. 4. Computational Complexity Analysis of the Proposed MSFCN
and Other Methods on the WHDLD Dataset.

App Results(Live Demo Using AWS EC2 Instance With Docker):
-------
![App Results](https://github.com/rajesh0305/MSFCN_Results/blob/main/app%20result.png)
![App Results](https://github.com/rajesh0305/MSFCN_Results/blob/main/app%20result1.png)
Fig. 5. App Results(Live Demo Using AWS EC2 Instance with Docker)
