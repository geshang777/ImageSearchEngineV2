from annoy import AnnoyIndex

# 用于保存annoy的模型信息
def store_annoy(feature_list,save_path):
    """
    feature list: list类型 feature的list
    save_path: 模型保存的位置
    """
    # 获得特征长度
    length = feature_list[0].shape[0]
    t = AnnoyIndex(length, 'angular')
    
    cnt = 0
    for feature in feature_list:
        t.add_item(cnt,feature)
        cnt +=1

    # 创建10颗二叉树
    t.build(10)
    t.save(save_path)


def get_scores(query,annoy_path,img_paths):
    """
    query: 检索图片的feature
    annoy_path: cls对应模型的位置，对应上面函数的save_path
    img_path: 所有图片路径按顺序保存的list，对应于代码里从kd_tree中load出来的img_paths
    """
    length = query.shape[0]
    annoy_tree = AnnoyIndex(length, 'angular')
    annoy_tree.load(annoy_path)
    index,distance = annoy_tree.get_nns_by_vector(query,20, include_distances=True)
    scores = [(dis,img_paths[idx]) for idx,dis in zip(index,distance) if dis>0.7]
    return scores