import os, csv, tqdm, shutil
import numpy as np
from PIL import Image
from sklearn import metrics
from itertools import islice
import torch, torchvision
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tensorboardX import SummaryWriter

class DatasetCFP(Dataset):
    def __init__(self, root, data_file, mode='train'):
        self.data_list = self.get_files(root, data_file=data_file)
        if mode == 'train':
            self.transforms= T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.299,0.224,0.225])])
        else:
            self.transforms = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.299,0.224,0.225])])

    def get_files(self, root, data_file):
        csv_reader = csv.reader(open(data_file))
        img_list = []
        for line in islice(csv_reader, 1, None): # Skip the header line and each line is a list of [image_path, label]
            img_list.append([os.path.join(root,line[0]), int(line[1])])
        return img_list

    def __getitem__(self, index):
        image_file,label = self.data_list[index]
        img = Image.open(image_file).convert("RGB")
        img_tensor = self.transforms(img)
        return img_tensor, label, image_file

    def __len__(self):
        return len(self.data_list)

class ResUnNet(nn.Module):
    def __init__(self,num_classes=1):
        super(ResUnNet, self).__init__()
        resnet_img = torchvision.models.resnet50(pretrained=True)  # pretrained ImageNet ResNet-34
        modules_img = list(resnet_img.children())[:-2]
        self.resnet_img = nn.Sequential(*modules_img)
        self.avgpool_fun = nn.AdaptiveAvgPool2d((1,1))  #
        self.affine_classifier = nn.Linear(2048, num_classes)
        
    def forward(self, image):
        out_img = self.resnet_img(image)
        avg_feature = self.avgpool_fun(out_img)
        avg_feature = torch.flatten(avg_feature, 1)
        result = self.affine_classifier(avg_feature)
        return result

def net_builder(name, num_classes=9):
    if name == 'ResUnNet50':
        net= ResUnNet(num_classes=num_classes)
    else:
        raise NameError("Unknow Model Name!")
    return net

def KL(alpha, c, device):
    beta = torch.ones((1, c)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha); dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step, device):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c, device)
    return (A + B)

def adjust_learning_rate(opt, optimizer, epoch):
    assert opt.lr_mode in ['step', 'poly', 'normal'], 'lr_mode not supported {}'.format(opt.lr_mode)
    step_mode = lambda epoch: opt.lr * (0.1 ** (epoch // opt.step))
    poly_mode = lambda epoch: opt.lr * (1 - epoch / opt.num_epochs) ** 0.9
    normal_mode = lambda epoch: opt.lr
    lr = {'step': step_mode, 'poly': poly_mode, 'normal': normal_mode}[opt.lr_mode](epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

############################################################################################################################################################################

def train(args=None,writer=None):
    train_loader = DataLoader(DatasetCFP(root=args.root, mode='train', data_file=args.train_file), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(DatasetCFP(root=args.root, mode='val', data_file=args.val_file), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(DatasetCFP(root=args.root, mode='test', data_file=args.test_file), batch_size=args.batch_size, shuffle=False, pin_memory=True)
    device = torch.device('cuda:{}'.format(args.cuda))
    model = net_builder(args.net_work, args.num_classes).to(device)
    print('Model have been loaded!,you chose the ' + args.net_work + '!')
    if args.trained_model_path:
        print("=> loading trained model '{}'".format(args.trained_model_path))
        checkpoint = torch.load(args.trained_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Done!')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)
    def val(val_dataloader, model, epoch, args, mode, device):
        print('\n')
        print('====== Start {} ======!'.format(mode))
        model.eval()
        labels = []
        outputs = []
        predictions = []
        gts = []
        correct = 0.0
        num_total = 0
        tbar = tqdm.tqdm(val_dataloader, desc='\r')
        with torch.no_grad():
            for i, img_data_list in enumerate(tbar):
                Fundus_img = img_data_list[0].to(device)
                cls_label = img_data_list[1].long().to(device)
                pred = model.forward(Fundus_img)
                evidences = [F.softplus(pred)]
                alpha = dict()
                alpha[0] = evidences[0] + 1
                S = torch.sum(alpha[0], dim=1, keepdim=True)
                E = alpha[0] - 1
                b = E / (S.expand(E.shape))
                pred = torch.softmax(b,dim=1)
                data_bach = pred.size(0)
                num_total += data_bach
                one_hot = torch.zeros(data_bach, args.num_classes).to(device).scatter_(1, cls_label.unsqueeze(1), 1)
                pred_decision = pred.argmax(dim=-1)
                for idx in range(data_bach):
                    outputs.append(pred.cpu().detach().float().numpy()[idx])
                    labels.append(one_hot.cpu().detach().float().numpy()[idx])
                    predictions.append(pred_decision.cpu().detach().float().numpy()[idx])
                    gts.append(cls_label.cpu().detach().float().numpy()[idx])
        epoch_auc = metrics.roc_auc_score(labels, outputs)
        Acc = metrics.accuracy_score(gts, predictions)
        if not os.path.exists(os.path.join(args.save_model_path, "{}".format(args.net_work))):
            os.makedirs(os.path.join(args.save_model_path, "{}".format(args.net_work)))
        with open(os.path.join(args.save_model_path,"{}/{}_Metric.txt".format(args.net_work,args.net_work)),'a+') as Txt:
            Txt.write("Epoch {}: {} == Acc: {}, AUC: {}\n".format(epoch,mode, round(Acc,6),round(epoch_auc,6)))
        print("Epoch {}: {} == Acc: {}, AUC: {}\n".format(epoch,mode,round(Acc,6),round(epoch_auc,6)))
        torch.cuda.empty_cache()
        return epoch_auc,Acc
    def train(train_loader, val_loader, test_loader, model, optimizer, criterion,writer,args,device):
        step = 0
        best_auc = 0.0
        best_auc_Test = 0.0
        for epoch in range(0,args.num_epochs+1):
            model.train()
            labels = []
            outputs = []
            tq = tqdm.tqdm(total=len(train_loader) * args.batch_size)
            tq.set_description('Epoch %d, lr %f' % (epoch, args.lr))
            loss_record = []
            train_loss = 0.0
            for i, img_data_list in enumerate(train_loader):
                Fundus_img = img_data_list[0].to(device)
                cls_label = img_data_list[1].long().to(device)
                optimizer.zero_grad()
                pretict = model(Fundus_img)
                evidences = [F.softplus(pretict)]
                loss_un = 0
                alpha = dict()
                alpha[0] = evidences[0] + 1
                S = torch.sum(alpha[0], dim=1, keepdim=True)
                E = alpha[0] - 1
                b = E / (S.expand(E.shape))
                Tem_Coef = epoch*(0.99/args.num_epochs)+0.01
                loss_CE = criterion(b/Tem_Coef, cls_label)
                loss_un += ce_loss(cls_label, alpha[0], args.num_classes, epoch, args.num_epochs, device)
                loss_ACE = torch.mean(loss_un)
                loss = loss_CE+loss_ACE
                loss.backward()
                optimizer.step()
                tq.update(args.batch_size)
                train_loss += loss.item()
                tq.set_postfix(loss='%.6f' % (train_loss / (i + 1)))
                step += 1
                one_hot = torch.zeros(pretict.size(0), args.num_classes).to(device).scatter_(1, cls_label.unsqueeze(1), 1)
                pretict = torch.softmax(pretict, dim=1)
                for idx_data in range(pretict.size(0)):
                    outputs.append(pretict.cpu().detach().float().numpy()[idx_data])
                    labels.append(one_hot.cpu().detach().float().numpy()[idx_data])
                if step%10==0:
                    writer.add_scalar('Train/loss_step', loss, step)
                loss_record.append(loss.item())
            tq.close()
            torch.cuda.empty_cache()
            loss_train_mean = np.mean(loss_record)
            epoch_train_auc = metrics.roc_auc_score(labels, outputs)
            del labels,outputs
            writer.add_scalar('Train/loss_epoch', float(loss_train_mean), epoch)
            writer.add_scalar('Train/train_auc', float(epoch_train_auc), epoch)
            print('loss for train : {}, {}'.format(loss_train_mean,round(epoch_train_auc,6)))
            if epoch % args.validation_step == 0:
                if not os.path.exists(os.path.join(args.save_model_path, "{}".format(args.net_work))):
                    os.makedirs(os.path.join(args.save_model_path, "{}".format(args.net_work)))
                with open(os.path.join(args.save_model_path, "{}/{}_Metric.txt".format(
                        args.net_work,args.net_work)), 'a+') as f:
                    f.write('EPOCH:' + str(epoch) + ',')
                mean_AUC, mean_ACC = val(val_loader, model, epoch,args,mode="val",device=device)
                writer.add_scalar('Valid/Mean_val_AUC', mean_AUC, epoch)
                best_auc = max(best_auc, mean_AUC)
                checkpoint_dir = os.path.join(args.save_model_path)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                mean_AUC_Test, mean_ACC_Test = val(test_loader, model, epoch, args, mode="Test",device=device)
                writer.add_scalar('Test/Mean_Test_AUC', mean_AUC_Test, epoch)
                print('===> Saving models...')
                def save_checkpoint_epoch(state, pred_AUC, pred_ACC, test_AUC, test_ACC, epoch, is_best, checkpoint_path, stage="val", filename='./checkpoint/checkpoint.pth.tar'):
                    torch.save(state, filename)
                    print('===> Saving models...') if is_best else None
                    shutil.copyfile(filename, os.path.join(checkpoint_path, 'model_{}_{:03d}_Val_{:.6f}_{:.6f}_Test_{:.6f}_{:.6f}.pth.tar'.format(stage,(epoch + 1),pred_AUC,pred_ACC,test_AUC,test_ACC))) if is_best else None
                save_checkpoint_epoch({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'mean_AUC': mean_AUC, 'mean_ACC': mean_ACC, 'mean_AUC_Test': mean_AUC_Test, 'mean_ACC_Test': mean_ACC_Test}, 
                                    mean_AUC, mean_ACC, mean_AUC_Test, mean_ACC_Test, epoch, True, checkpoint_dir, stage="Test", filename=os.path.join(checkpoint_dir,"checkpoint.pth.tar"))
    train(train_loader, val_loader, test_loader, model, optimizer, criterion,writer,args,device)
    
def thresholding(args=None):
    args.net_work = "ResUnNet50"
    args.trained_model_path = './Trained/UIOS.pth.tar'
    device = torch.device('cuda:{}'.format(args.cuda))
    args.device = device
    model = net_builder(args.net_work, args.num_classes).to(device)
    print('Model have been loaded!,you chose the ' + args.net_work + '!')
    print("=> loading trained model '{}'".format(args.trained_model_path))
    checkpoint = torch.load(args.trained_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print('Done!')
    test_loader = DataLoader(DatasetCFP(root=args.root, mode='test', data_file=args.val_file), batch_size=args.batch_size, shuffle=False, pin_memory=True)
    def val(val_dataloader, model, args, mode, device):
        print('\n')
        print('====== Start {} ======!'.format(mode))
        model.eval()
        u_list = []
        u_label_list = []
        tbar = tqdm.tqdm(val_dataloader, desc='\r') 
        with torch.no_grad():
            for i, img_data_list in enumerate(tbar):
                Fundus_img = img_data_list[0].to(device)
                cls_label = img_data_list[1].long().to(device)
                pred = model.forward(Fundus_img)
                evidences = [F.softplus(pred)]
                alpha = dict()
                alpha[0] = evidences[0] + 1
                S = torch.sum(alpha[0], dim=1, keepdim=True)
                E = alpha[0] - 1
                b = E / (S.expand(E.shape))
                u = args.num_classes / S
                un_gt = 1 - torch.eq(b.argmax(dim=-1), cls_label).float()
                data_bach = pred.size(0)
                for idx in range(data_bach):
                    u_list.append(u.cpu()[idx].numpy())
                    u_label_list.append(un_gt.cpu()[idx].numpy())
        return u_list, u_label_list
    u_list, u_label_list = val(test_loader, model, args, mode="Validation", device=device) 
    fpr_Pri, tpr_Pri, thresh = metrics.roc_curve(u_label_list, u_list)
    max_j = max(zip(fpr_Pri, tpr_Pri), key=lambda x: 2*x[1] - x[0])
    pred_thresh = thresh[list(zip(fpr_Pri, tpr_Pri)).index(max_j)]
    print("opt_pred ===== {}".format(pred_thresh))

def inference(args=None):
    args.net_work = "ResUnNet50"
    args.trained_model_path = './Trained/UIOS.pth.tar'
    device = torch.device('cuda:{}'.format(args.cuda))
    args.device = device
    model = net_builder(args.net_work, args.num_classes).to(device)
    print('Model have been loaded!,you chose the ' + args.net_work + '!')
    print("=> loading trained model '{}'".format(args.trained_model_path))
    checkpoint = torch.load(args.trained_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print('Done!')
    Thres = 0.1158
    Results_Heads = ["Imagefiles", 'Prediction results','Reliability']
    args.root = "./Datasets/OOD_Test"
    csv_file = "./Datasets/Pred_test.csv"
    test_loader = DataLoader(DatasetCFP(root=args.root, mode='test', data_file="Datasets/{}.csv".format(csv_file)), batch_size=args.batch_size, shuffle=False, pin_memory=True)
    def val(val_dataloader, model, args, Normal_average=0.0,device=None):
        print('\n')
        model.eval()
        tbar = tqdm.tqdm(val_dataloader, desc='\r')
        All_Infor = []
        with torch.no_grad():
            for i, img_data_list in enumerate(tbar):
                Fundus_img = img_data_list[0].to(device)
                image_files = img_data_list[2] # is a list of image files path
                pred = model.forward(Fundus_img)
                evidences = [F.softplus(pred)]
                alpha = dict()
                alpha[0] = evidences[0] + 1
                S = torch.sum(alpha[0], dim=1, keepdim=True)
                E = alpha[0] - 1
                b = E / (S.expand(E.shape))
                u = args.num_classes / S
                batch = b.shape[0]
                pred_con = pred.argmax(dim=-1)
                for idx_bs_u in range(batch):
                    if u[idx_bs_u] >= Normal_average:
                        All_Infor.append(['/'.join(image_files[idx_bs_u].split('/')[-2:]), pred_con.cpu().detach().float().numpy()[idx_bs_u], "Unreliable"])
                    else:
                        All_Infor.append(['/'.join(image_files[idx_bs_u].split('/')[-2:]), pred_con.cpu().detach().float().numpy()[idx_bs_u], "Reliable"])
        return All_Infor
    Results_Contents = val(test_loader, model, args, Normal_average=Thres, device=device)
    with open("PredictionResults/Results.csv", 'w', newline='') as f:
        writer = csv.writer(f); writer.writerow(Results_Heads); writer.writerows(Results_Contents)
        
class DefaultConfig(object):
    root = "/raid/DTS/Dataset"; train_file = "Datasets/train.csv"; val_file = "Datasets/val.csv"; test_file = "Datasets/test.csv"; log_dirs = './Logs_Adam_0304'; 
    net_work = 'ResUnNet50'; num_classes = 9; num_epochs = 100; batch_size = 64; validation_step = 1; lr = 1e-4; lr_mode = 'poly'; momentum = 0.9; weight_decay = 1e-4; 
    pretrained = False; pretrained_model_path = None; cuda = 0; num_workers = 4; use_gpu = True; 
    trained_model_path = ''; predict_fold = 'predict_mask'; save_model_path = './Model_Saved'.format(net_work,lr)
        
if __name__ == '__main__':
    args = DefaultConfig()
    writer = SummaryWriter(log_dir=args.log_dirs)
    train(args=args, writer=writer)
    # inference(args=args)
    # thresholding(args=args)