import os.path
import glob
from utils import load_file,save_pkl,load_pkl

class DataProcesser(object):
    @staticmethod
    def get_labels():
        tokens_list = []
        txt_files = glob.glob('D:/test_BiLSTM_CRF/data/*.bioes')
        base_path = os.path.abspath(os.path.join(os.path.pardir))
        label_path = os.path.join(base_path,'output','label_list.pkl')
        if os.path.exists(label_path):
            labels = load_pkl(label_path)
        else:
            for file in txt_files:
                tokens_list.extend(load_file(file,sep=" "))
            labels = set([tokens[1] for tokens in tokens_list if len(tokens) == 2])

        if len(labels) == 0:
            print("ERROR")
        else:
            save_pkl(labels,label_path)
        return labels

    @staticmethod
    def get_label2id_id2label(label_list):
        base_path = os.path.abspath(os.path.join(os.path.pardir))
        label2id_path = os.path.join(base_path,'output',"label2id.pkl")
        if os.path.exists(label2id_path):
            label2id = load_pkl(label2id_path)
            print(label2id)
        else:
            label2id = {l: i for i, l in enumerate(label_list)}
            save_pkl(label2id,label2id_path)
            #with open(label2id_path, 'w', encoding='utf-8') as fout:
            #    for sig in label2id:
            #        fout.write(sig)

        id2label = {value: key for key, value in label2id.items()}
        return label2id, id2label

    def get_dataset(self, config: Config, tokenizer, mode="train"):
        """
        对指定数据集进行预处理，进一步封装数据，包括:
        examples：[InputExample(guid=index, text=text, label=label)]
        features：[InputFeatures( input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask,
                                  label_id=label_ids,
                                  ori_tokens=ori_tokens)]
        data： 处理完成的数据集, TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_label_ids)
        """
        if mode == "train":
            file_path = config.train_file
        elif mode == "eval":
            file_path = config.eval_file
        elif mode == "test":
            file_path = config.test_file
        else:
            raise ValueError("mode must be one of train, eval, or test")

        # 读取输入数据，进一步封装
        examples = self.get_input_examples(file_path, separator=config.sep)

        # 对输入数据进行特征转换
        features = self.convert_examples_to_features(config, examples, tokenizer)

        # 获取全部数据的特征，封装成TensorDataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_label_ids)

        return examples, features, data