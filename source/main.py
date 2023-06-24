import torch.cuda

from transformers import AdamW,get_linear_schedule_with_warmup,BertConfig,BertTokenizer
from config import Config
from data_processer import DataProcesser
from logger import logger as logger
from models import BERT_BiLSTM_CRF


class NerMain(object):
    def __init__(self):
        self.processer = DataProcesser()
        self.config = Config()

    def train(self):
        "模型训练"
        use_gpu = torch.cuda.is_available() and self.config.use_gpu
        device = torch.device('cuda' if use_gpu else self.config.device)
        self.config.device = device
        n_gpu = torch.cuda.device_count()
        logger.info(f"available device: {device}，count_gpu: {n_gpu}")
        logger.info("====================== Start Data Pre-processing ======================")
        # 读取训练数据获取标签
        label_list = self.processer.get_labels()
        num_labels = len(label_list)
        logger.info(f"loading labels successful! the size is {num_labels}, label is: {','.join(list(label_list))}")

        #获取label2id、id2label的映射
        label2id, id2label = self.processer.get_label2id_id2label(label_list)
        logger.info("loading label2id and id2label dictionary successful!")

        if self.config.do_train:
            # 初始化tokenizer(标记生成器)、bert_config、BERT_BiLSTM_CRF、加载
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=self.config.do_lower_case)
            bert_config = BertConfig.from_pretrained('bert-base-chinese', num_labels=num_labels)
            model = BERT_BiLSTM_CRF.from_pretrained('bert-base-chinese', config=bert_config,
                                                    need_birnn=self.config.need_birnn, rnn_dim=self.config.rnn_dim)

            model.to(device)
            logger.info("loading tokenizer、bert_config and bert_bilstm_crf model successful!")

            if use_gpu and n_gpu > 1:
                model = torch.nn.DataParallel(model)

            logger.info("starting load train data and data_loader...")
            # 获取训练样本、样本特征、TensorDataset信息
            train_examples, train_features, train_data = self.processor.get_dataset(self.config, tokenizer, mode="train")

            # 训练数据载入
            train_data_loader = DataLoader(train_data, batch_size=self.config.train_batch_size, sampler=RandomSampler(train_data))
            logger.info("loading train data_set and data_loader successful!")

            eval_examples, eval_features, eval_data = [], [], None
            if self.config.do_eval:
                logger.info("starting load eval data...")
                eval_examples, eval_features, eval_data = self.processor.get_dataset(self.config, tokenizer, mode="eval")
                logger.info("loading eval data_set successful!")
            logger.info("====================== End Data Pre-processing ======================")

            # 初始化模型参数优化器
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)

            # 初始化学习率优化器
            t_total = len(train_data_loader) // self.config.gradient_accumulation_steps * self.config.num_train_epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=t_total)
            logger.info("loading AdamW optimizer、Warmup LinearSchedule and calculate optimizer parameter successful!")

            logger.info("====================== Running training ======================")
            logger.info(
                f"Num Examples:  {len(train_data)}, Num Batch Step: {len(train_data_loader)}, "
                f"Num Epochs: {self.config.num_train_epochs}, Num scheduler steps：{t_total}")

            # 启用 BatchNormalization 和 Dropout
            model.train()
            global_step, tr_loss, logging_loss, best_f1 = 0, 0.0, 0.0, 0.0
            for ep in trange(int(self.config.num_train_epochs), desc="Epoch"):
                logger.info(f"########[Epoch: {ep}/{int(self.config.num_train_epochs)}]########")
                model.train()
                for step, batch in enumerate(tqdm(train_data_loader, desc="DataLoader")):
                    logging.info(f"####[Step: {step}/{len(train_data_loader)}]####")

                    batch = tuple(t.to(device) for t in batch)
                    input_ids, token_type_ids, attention_mask, label_ids = batch
                    outputs = model(input_ids, label_ids, token_type_ids, attention_mask)
                    loss = outputs

                    if use_gpu and n_gpu > 1:
                        # mean() to average on multi-gpu.
                        loss = loss.mean()

                    if self.config.gradient_accumulation_steps > 1:
                        loss = loss / self.config.gradient_accumulation_steps

                    # 反向传播
                    loss.backward()
                    tr_loss += loss.item()

                    # 优化器_模型参数的总更新次数，和上面的t_total对应
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        # 更新参数
                        optimizer.step()
                        scheduler.step()
                        # 梯度清零
                        model.zero_grad()
                        global_step += 1

                        if self.config.logging_steps > 0 and global_step % self.config.logging_steps == 0:
                            tr_loss_avg = (tr_loss - logging_loss) / self.config.logging_steps
                            writer.add_scalar("Train/loss", tr_loss_avg, global_step)
                            logging_loss = tr_loss
                # 模型验证

                if self.config.do_eval:
                    logger.info("====================== Running Eval ======================")
                    all_ori_tokens_eval = [f.ori_tokens for f in eval_features]
                    overall, by_type = self.evaluate(self.config, eval_data, model, id2label, all_ori_tokens_eval)

                    # add eval result to tensorboard
                    f1_score = overall.fscore
                    writer.add_scalar("Eval/precision", overall.prec, ep)
                    writer.add_scalar("Eval/recall", overall.rec, ep)
                    writer.add_scalar("Eval/f1_score", overall.fscore, ep)

                    # save the best performs model
                    if f1_score > best_f1:
                        logging.info(f"******** the best f1 is {f1_score}, save model !!! ********")
                        best_f1 = f1_score
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, 'module') else model
                        model_to_save.save_pretrained(self.config.output_path)
                        tokenizer.save_pretrained(self.config.output_path)

                        # Good practice: save your training arguments together with the trained model
                        torch.save(self.config, os.path.join(self.config.output_path, 'training_config.bin'))
                        torch.save(model, os.path.join(self.config.output_path, 'ner_model.ckpt'))
                        logging.info("training_args.bin and ner_model.ckpt save successful!")
            writer.close()
            logging.info("NER model training successful!!!")
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    NerMain().train()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
