from model import *
from utils import EarlyStopping
import wandb
import math
class Trainer:
    def __init__(self, action, version, section_num, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, dataset, split):
        self.version = int(version)

        self.section_num = section_num
        self.models = []
        if version == 1:
            for i in range(section_num):
                self.models.append(MultiStageModel(num_layers_PG, num_layers_R, num_f_maps, dim, num_classes))
        elif version == 2:
            for i in range(section_num):
                self.models.append(MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes))
        # if action == 'train':
        #     for i in range(section_num):
        #         wandb.watch(self.models[i])
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, section_num, num_epochs, batch_size, learning_rate, device):
        wandb.watch(self.models[section_num])
        self.models[section_num].to(device)
        self.models[section_num].train()
        optimizer = optim.Adam(self.models[section_num].parameters(), lr=learning_rate)
        early_stopping = EarlyStopping(patience=5, verbose=True, path=save_dir)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size,section_num)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.models[section_num](batch_input, mask)

                loss = 0
                for p in predictions:
                    loss_tmp = p.transpose(2, 1).contiguous().view(-1, self.num_classes)
                    loss += self.ce(loss_tmp, batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                wandb.log({"loss": loss})
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                each_correct = (predicted == batch_target).float()*mask[:, 0, :].squeeze(1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            # torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            if (epoch>89 or (epoch+1)%10==0):
                torch.save(self.models[section_num].state_dict(), save_dir + "/epoch-" + str(epoch + 1) + "_"+str(section_num)+ ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + "_"+str(section_num) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct) / total))
            # early_stopping(epoch_loss, self.model)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            # print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
            #                                                    float(correct) / total))
        # self.model.load_state_dict(torch.load(save_dir+'/checkpoint.pt'))
        # torch.save(self.model.state_dict(), save_dir + "/bestmodel.model")
        # torch.save(optimizer.state_dict(), save_dir + "/bestmodel.opt")


    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        for sec in range(self.section_num):
            self.models[sec].eval()
            with torch.no_grad():
                self.models[sec].to(device)
                self.models[sec].load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + "_"+str(sec)+".model"))
                # self.model.load_state_dict(torch.load(model_dir + "/bestmodel.model"))
                file_ptr = open(vid_list_file, 'r')
                list_of_vids = file_ptr.read().split('\n')[:-1]
                file_ptr.close()
                for vid in list_of_vids:
                    print(vid)
                    features = np.load(features_path + vid.split('.')[0] + '.npy')
                    features = features[:, ::sample_rate]
                    video_length = np.shape(features)[1]
                    section = math.ceil(video_length / self.section_num)
                    features = features[:, sec * section:(sec + 1) * section]
                    if sec == (self.section_num-1):
                        if (sec + 1) * section < video_length:
                            assert('Error')

                    input_x = torch.tensor(features, dtype=torch.float)
                    input_x.unsqueeze_(0)
                    input_x = input_x.to(device)
                    predictions = self.models[sec](input_x, torch.ones(input_x.size(), device=device))
                    _, predicted = torch.max(predictions[-1].data, 1)
                    predicted = predicted.squeeze()
                    recognition = []
                    for i in range(len(predicted)):
                        recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                    f_name = vid.split('/')[-1].split('.')[0]
                    if sec == 0:
                        f_ptr = open(results_dir + "/" + f_name, "w")
                        f_ptr.write("### Frame level recognition2: ###\n")
                    else:
                        f_ptr = open(results_dir + "/" + f_name, "a")
                        f_ptr.write(' ')
                    f_ptr.write(' '.join(recognition))
                    f_ptr.close()