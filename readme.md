## Grayscale  &&  RGB image
    1. Config里面
    ```
    IMAGE_CHANNEL_COUNT
    MEAN_PIXEL
    ```
    2. Dataset.load_image函数
    cv2.imread(**, 0) 读入单通道image数据，然后np.expand_dims

    3. load pretrained coco weights的时候
    要additonal exclude掉第一层权重，因为INPUT_SHAPE不一样
<<<<<<< HEAD
    如果RPN_ANCHOR相关参数改了，还要additonal exclude掉'rpn_model'层
=======
>>>>>>> c1554cb631775375c34ef017ddabe751994e2c1d

    4. visualize.display_instances
    因为plt.show需要三通道的输入，所以先把灰度图转成3通道
    ```
    # convert single channel image to 3-channels
    if image.shape[-1] == 1:
        image = np.concatenate((image, image, image), axis=2)
    ```

    5. eval.py中modellib.load_image_gt会调用Dataset的load_image函数函数
    确保前后一致


## Single class  &&  multi- classes
    1. Config里面
    ```
    NUM_CLASSES
    ```
    2. load_mask函数
    cv2.imread(**, 0) 读入单通道mask数据，然后np.expand_dims


## Single object  &&  multi- objects
    1. load_mask函数
    mask必须是W*H*COUNT，single class -> W*H*1


## Dataset
    1. 有mask
        参考mytrain的脚本
    2. 有json
        参考原工程的Samples


## config参数
    1. RPN_ANCHOR_SCALES
    这个东西要与网络的backbone保持一致，默认ResNet101-FPN结构，详见model.build
    RPN用了五个尺度的特征层，因此RPN_ANCHOR_SCALES的shape必须有五个元素


<<<<<<< HEAD
## anchors聚类
    数据集中mask尺寸比较特殊，用yolo_kmeans跑了下聚类
    相应要修改config中的RPN_ANCHOR_SCALES和RPN_ANCHOR_RATIOS
    load pretrained coco weights的时候，要additonal exclude掉'rpn_model'层


## warning
    "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
    [参考](https://github.com/matterport/Mask_RCNN/issues/749)


## todoList
    1. model.train: 
    custom callbacks，不知道有哪些在训练过程中值得关注的指标
    2. model
    custom losses: unbalance for class branch & mixed loss for mask branch


    

=======
    
>>>>>>> c1554cb631775375c34ef017ddabe751994e2c1d
