# 加入校准延迟量的补偿量部分
# 把第一层改为相同的C和S
from net_parts import *

print('main 成功运行 \n')
# 读取测量数据
root = r"C:\Users\98072\ysl_file\PITUNN_code" # 运行前修改根目录
file_path = root + r"\MatlabCode\data.csv"
cos_delta_2_np = read_col(file_path, 'cos_delta_2')
cos_delta_1Minus2_np = read_col(file_path, 'cos_delta_1Minus2')
sin_delta_1Minus2_np = read_col(file_path, 'sin_delta_1Minus2')
cos_delta_1Plus2_np = read_col(file_path, 'cos_delta_1Plus2')
sin_delta_1Plus2_np = read_col(file_path, 'sin_delta_1Plus2')
cos_delta_1_np = read_col(file_path, 'cos_delta_1')
sin_delta_1_np = read_col(file_path, 'sin_delta_1')
hyper = torch.from_numpy(np.array(pd.read_csv(root+r"\MatlabCode\hyper.csv"), dtype=np.float32)).to('cuda:0')
Y_np = read_col(file_path, 'Y')
M_basis_np = np.array(pd.read_csv
                      (root + r"\MatlabCode\M_basis.csv"),
                      dtype=np.float32)
# 将数据转换为GPU张量
cos_delta_2 = torch.from_numpy(cos_delta_2_np).unsqueeze(0).to('cuda:0')
cos_delta_1Minus2 = torch.from_numpy(cos_delta_1Minus2_np).unsqueeze(0).to('cuda:0')
sin_delta_1Minus2 = torch.from_numpy(sin_delta_1Minus2_np).unsqueeze(0).to('cuda:0')
cos_delta_1Plus2 = torch.from_numpy(cos_delta_1Plus2_np).unsqueeze(0).to('cuda:0')
sin_delta_1Plus2 = torch.from_numpy(sin_delta_1Plus2_np).unsqueeze(0).to('cuda:0')
cos_delta_1 = torch.from_numpy(cos_delta_1_np).unsqueeze(0).to('cuda:0')
sin_delta_1 = torch.from_numpy(sin_delta_1_np).unsqueeze(0).to('cuda:0')
Y = torch.from_numpy(Y_np).unsqueeze(0).to('cuda:0')
M_basis = torch.from_numpy(M_basis_np).unsqueeze(0).to('cuda:0')

N_simple, C_simple, S_simple, I_simple, running_loss_1, time_cost_1 = \
    simpleNCS(M_basis, Y, cos_delta_2, cos_delta_1Minus2, cos_delta_1Plus2, sin_delta_1Minus2, sin_delta_1Plus2,
              layers=4, hidden=400, num=2, num2=200, f_set=hyper[0], beta=hyper[1], gama=hyper[2], range1=0.1, range2=0.9, seed=1)

e1, e2, e3, e4, I_epsilon, running_loss_2, time_cost_2 = \
    epsilonNCS(Y, cos_delta_2, cos_delta_1Minus2, cos_delta_1Plus2, sin_delta_1Minus2, sin_delta_1Plus2,
               cos_delta_1, sin_delta_1, N_simple.detach(), C_simple.detach(), S_simple.detach(),
               layers=2, hidden=400, num=2, num2=100, range1=0.1, range2=0.9, seed=1)

e = 1
N_generate, C_generate, S_generate, I_generate, running_loss_3, time_cost_3 = \
    outNCS(M_basis, Y, cos_delta_2, cos_delta_1Minus2, cos_delta_1Plus2, sin_delta_1Minus2, sin_delta_1Plus2,
           cos_delta_1, sin_delta_1,
           e * e1.detach(), e * e2.detach(), e * e3.detach(), e * e4.detach(),
           layers=2, hidden=400, num=2, num2=300, f_set=hyper[3], beta=hyper[4], gama=hyper[5], range1=0.1, range2=0.9, seed=1)

# 储存数据
out_N_simple = N_simple.cpu().squeeze(0).squeeze(0).detach().numpy()
out_C_simple = C_simple.cpu().squeeze(0).squeeze(0).detach().numpy()
out_S_simple = S_simple.cpu().squeeze(0).squeeze(0).detach().numpy()
out_N = N_generate.cpu().squeeze(0).squeeze(0).detach().numpy()
out_C = C_generate.cpu().squeeze(0).squeeze(0).detach().numpy()
out_S = S_generate.cpu().squeeze(0).squeeze(0).detach().numpy()
out_I = I_generate.cpu().squeeze(0).squeeze(0).detach().numpy()
out_e1 = e1.cpu().squeeze(0).squeeze(0).detach().numpy()
out_e2 = e2.cpu().squeeze(0).squeeze(0).detach().numpy()
out_e3 = e3.cpu().squeeze(0).squeeze(0).detach().numpy()
out_e4 = e4.cpu().squeeze(0).squeeze(0).detach().numpy()
x = pd.DataFrame({'out_N': out_N, 'out_C': out_C,
                  'out_S': out_S, 'I_predict': out_I,
                  'out_N_simple': out_N_simple, 'out_C_simple': out_C_simple,
                  'out_S_simple': out_S_simple})
x.to_csv(root + r"\MatlabCode\result.csv")
running_loss = np.append(running_loss_1, time_cost_3)
running_loss = np.append(running_loss, out_e1)
running_loss = np.append(running_loss, out_e2)
running_loss = np.append(running_loss, out_e3)
running_loss = np.append(running_loss, out_e4)
y = pd.DataFrame({'Running_loss': running_loss})
y.to_csv(root + r"\MatlabCode\loss.csv")
