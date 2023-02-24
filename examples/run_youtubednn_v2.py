import sys
sys.path.append("/home/hdp-portrait/git/DeepMatch")

import pandas as pd
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss, NegativeSampler
from preprocess import *
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model

if __name__ == "__main__":

    data = pd.read_csvdata = pd.read_csv("./traindata_all.log",sep="\t")
    # data['genres'] = list(map(lambda x: x.split('|')[0], data['genres'].values))

    sparse_features = ["img_id",
                       "uid",
                       "serverprovince",
                       "servercity",
                       "city_level",
                       "nettype",
                       "model",
                       "os",
                       "osversion",
                       "brand"
                       ]
    SEQ_LEN = 50
    embedding_dim = 50

    # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`

    feature_max_idx = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    user_profile = data[["uid", "serverprovince", "servercity", "city_level", "nettype","model","os","osversion","brand"]].drop_duplicates('uid')

    item_profile = data[["img_id"]].drop_duplicates('img_id')

    user_profile.set_index("uid", inplace=True)

    user_item_list = data.groupby("uid")['img_id'].apply(list)

    train_set, test_set = gen_data_set_v2(data, SEQ_LEN, 0)

    train_model_input, train_label = gen_model_input_v2(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input_v2(test_set, user_profile, SEQ_LEN)

    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    user_feature_columns = [SparseFeat('uid', feature_max_idx['uid'], embedding_dim),
                            SparseFeat("serverprovince", feature_max_idx['serverprovince'], embedding_dim),
                            SparseFeat("servercity", feature_max_idx['servercity'], embedding_dim),
                            SparseFeat("city_level", feature_max_idx['city_level'], embedding_dim),
                            SparseFeat("nettype", feature_max_idx['nettype'], embedding_dim),
                            SparseFeat("model", feature_max_idx['model'], embedding_dim),
                            SparseFeat("os", feature_max_idx['os'], embedding_dim),
                            SparseFeat("osversion", feature_max_idx['osversion'], embedding_dim),
                            SparseFeat("brand", feature_max_idx['brand'], embedding_dim),
                            VarLenSparseFeat(SparseFeat('hist_img_id', feature_max_idx['img_id'], embedding_dim, embedding_name="img_id"), SEQ_LEN, 'mean', 'hist_len')
                            ]

    item_feature_columns = [SparseFeat('img_id', feature_max_idx['img_id'], embedding_dim)]

    from collections import Counter

    train_counter = Counter(train_model_input['img_id'])
    item_count = [train_counter.get(i, 0) for i in range(item_feature_columns[0].vocabulary_size)]
    sampler_config = NegativeSampler('frequency', num_sampled=5, item_name='img_id', item_count=item_count)

    # 3.Define Model and train

    import tensorflow as tf

    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    else:
        K.set_learning_phase(True)

    model = YoutubeDNN(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, embedding_dim), sampler_config=sampler_config)
    # model = MIND(user_feature_columns, item_feature_columns, dynamic_k=False, k_max=2, user_dnn_hidden_units=(64, embedding_dim), sampler_config=sampler_config)

    model.compile(optimizer="adam", loss=sampledsoftmaxloss)

    history = model.fit(train_model_input, train_label, batch_size=256, epochs=10, verbose=1, validation_split=0.0, )

    # 4. Generate user features for testing and full item features for retrieval
    test_user_model_input = test_model_input
    all_item_model_input = {"img_id": item_profile['img_id'].values}

    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    print(user_embs.shape)
    print(item_embs.shape)

    # 5. [Optional] ANN search by faiss  and evaluate the result

    import heapq
    from collections import defaultdict
    from tqdm import tqdm
    import numpy as np
    import faiss
    from deepmatch.utils import recall_N

    k_max = 1
    topN = 50
    test_true_label = {line[0]: [line[1]] for line in test_set}

    index = faiss.IndexFlatIP(embedding_dim)
    # faiss.normalize_L2(item_embs)
    index.add(item_embs)
    # faiss.normalize_L2(user_embs)

    if len(user_embs.shape) == 2:  # multi interests model's shape = 3 (MIND,ComiRec)
        user_embs = np.expand_dims(user_embs, axis=1)

    score_dict = defaultdict(dict)
    for k in range(k_max):
        user_emb = user_embs[:, k, :]
        D, I = index.search(np.ascontiguousarray(user_emb), topN)
        for i, uid in tqdm(enumerate(test_user_model_input['uid']), total=len(test_user_model_input['uid'])):
            if np.abs(user_emb[i]).max() < 1e-8:
                continue
            for score, itemid in zip(D[i], I[i]):
                score_dict[uid][itemid] = max(score, score_dict[uid].get(itemid, float("-inf")))

    s = []
    hit = 0
    for i, uid in enumerate(test_user_model_input['uid']):
        pred = [item_profile['img_id'].values[x[0]] for x in
                heapq.nlargest(topN, score_dict[uid].items(), key=lambda x: x[1])]
        filter_item = None
        recall_score = recall_N(test_true_label[uid], pred, N=topN)
        s.append(recall_score)
        if test_true_label[uid] in pred:
            hit += 1

    print("recall", np.mean(s))
    print("hr", hit / len(test_user_model_input['uid']))
