import pandas as pd

from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import xgboost as xgb
import pickle

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, pre_data=None):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'Sum':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            pre_data=pre_data,
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def calculate_mse(self, y_true, y_pred):
        # 均方误差
        mse = np.mean(np.abs(y_true - y_pred))
        return mse

    def predict(self, setting, load=False):
        pre_data = pd.read_csv('TestData.csv')
        # 数据都读取进来
        slresults = []
        offeredresults = []
        slreals = []
        offeredreals = []
        sllosss = []
        offeredlosss = []
        Fofferedreals = []
        Abandresults = []
        ASAresults = []
        ASAreals = []
        for i in range(len(pre_data) - 2):
            if i == 0:
                pred_data, pred_loader = self._get_data(flag='pred')
                slreals.append(pre_data['sl'][i])
                offeredreals.append(pre_data['offered'][i])
                Fofferedreals.append(pre_data['Foffered'][i])
                ASAreals.append(pre_data['Foffered'][i])
            else:
                pred_data, pred_loader = self._get_data(flag='pred', pre_data=pre_data[['date','sl','offered','ASA']].iloc[: i])
                slreals.append(pre_data['sl'][i])
                offeredreals.append(pre_data['offered'][i])
                Fofferedreals.append(pre_data['Foffered'][i])
                ASAreals.append(pre_data['Foffered'][i])
            print(f'预测第{i} 次')
            if load:
                path = os.path.join(self.args.checkpoints, setting)
                best_model_path = path+'/'+'checkpoint.pth'
                self.model.load_state_dict(torch.load(best_model_path))

            self.model.eval()

            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
                pred, true = self._process_one_batch(
                    pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                pred = pred_data.inverse_transform(pred)
                slresults.append(pred[0][0][2].detach().cpu().numpy())
                ASAresults.append(pred[0][0][1].detach().cpu().numpy())
                offeredresults.append(pred[0][0][0].detach().cpu().numpy())     
                print(pred)  
        slresults = [0.95 if x > 1 else 0.2 if x < 0 else x for x in slresults]               
        offeredMe = []
        slMe = []
        for i in range(len(slresults)):
            sllosss.append(self.calculate_mse(slresults[i], slreals[i]))
            offeredlosss.append(self.calculate_mse(offeredresults[i], offeredreals[i]))
            offeredMe.append(offeredresults[i] - Fofferedreals[i])
            slMe.append(slresults[i] - slreals[i])
            

        params = {
            'objective': 'multi:softmax',
            'num_class': 4,
            'max_depth': 13,
            'eta': 0.05,
            'eval_metric': 'merror'
        }
        
        # 生成df
        df = pd.DataFrame({'shift':pre_data['shift'],'truegroup':pre_data['truegroup'],'slreal':slreals,'slforecast':slresults,'ASAresults':ASAresults,'ASA':ASAreals,
                           'offeredreal':offeredreals,'offeredforecast':offeredresults,'offeredMe':offeredMe,},index=range(len(slresults)))
        df[['offeredforecast','ASAresults','slforecast']] = df[['offeredforecast','ASAresults','slforecast']].astype('float')
        df["shift"] = df["shift"].astype('category').cat.codes
        averages = []
        # 生成过去同一时间段的平均值
        numbers = list(range(50))
        for num in numbers:
            average = df[df['shift'] == num]['slreal'].mean()
            averages.append(average)
        df['slmean'] = [averages[i % 50] for i in range(len(df))]
        # 生成当天过去两天同一时间段的值
        df['cycle_index'] = (df.index % 50) + 1
        df['prev_day_value'] = df.groupby('cycle_index')['slreal'].shift(1)
        df['prev_two_days_value'] = df.groupby('cycle_index')['slreal'].shift(2)
        df['prev_three_days_value'] = df.groupby('cycle_index')['slreal'].shift(3)
        df['prev_offered_day_value'] = df.groupby('cycle_index')['offeredreal'].shift(1)
        df['prev_offered_two_days_value'] = df.groupby('cycle_index')['offeredreal'].shift(2)
        df['prev_offered_three_days_value'] = df.groupby('cycle_index')['offeredreal'].shift(3)

        # 使用平均值填充每列的 NaN 值
        test_data = df.dropna()
        def generate_random_value(label):
            if label == 0:
                return np.random.uniform(0.25, 0.5)
            elif label == 1:
                return np.random.uniform(0.5, 0.65)
            elif label == 2:
                return np.random.uniform(0.65, 0.8)
            elif label == 3:
                return np.random.uniform(0.8, 1)
            else:
                return None

        test_features = test_data[['shift', 'slmean', 'prev_day_value', 'prev_two_days_value','prev_three_days_value','prev_offered_day_value',
                 'prev_offered_two_days_value','prev_offered_three_days_value','slforecast', 'ASA','ASAresults',
                 'offeredreal','offeredforecast', 'offeredMe']]
        
        train_labels = test_data['truegroup']
        mask = (0.8 > test_features['slforecast']) & (test_features['slforecast'] > 0.5)
        test_feature = test_features[mask]
        dtrain = xgb.DMatrix(data=test_features, label=train_labels)
        dtest = xgb.DMatrix(data=test_feature)
        train = True
        if train:
            # 训练模型
            num_rounds = 25
            model = xgb.train(params, dtrain, num_rounds)
            with open('Xgboostmodels.pkl', 'wb') as file:
                pickle.dump(model, file)
        with open('Xgboostmodels.pkl', 'rb') as f:
            model = pickle.load(f)
            # 预测结果
        pred_labels = model.predict(dtest)
        print(pred_labels)
        pred_labels = pd.Series(pred_labels, index=test_feature.index)
        new_series = pred_labels.apply(generate_random_value)
        print(test_feature.index)
        df.loc[test_feature.index, 'slforecast'] = new_series
        
        df = pd.DataFrame({'slreal': slreals, 'slforecast': df['slforecast'], 'offeredreal': offeredreals,
                           'ASAresults':ASAresults, 'Foffered':Fofferedreals, 'offeredforecast': offeredresults,
                           'slloss': sllosss,'ASAresults' ,'offeredloss': offeredlosss,'offeredMe':offeredMe, 'slMe':slMe})
        df.to_csv('output.csv', index=False)

        return
        
    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y
