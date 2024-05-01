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

N_simple, C_simple, S_simple, I_simple, running_loss, time_cost = \
    DIPSP(M_basis, Y, cos_delta_2, cos_delta_1Minus2, cos_delta_1Plus2, sin_delta_1Minus2, sin_delta_1Plus2,
              num=2, num2=300, f_set=hyper[0], beta=hyper[1], range1=0.1, range2=0.9, seed=1)

# 储存数据
out_N = N_simple.cpu().squeeze(0).squeeze(0).detach().numpy()
out_C = C_simple.cpu().squeeze(0).squeeze(0).detach().numpy()
out_S = S_simple.cpu().squeeze(0).squeeze(0).detach().numpy()
out_I = I_simple.cpu().squeeze(0).squeeze(0).detach().numpy()
x = pd.DataFrame({'out_N': out_N, 'out_C': out_C,
                  'out_S': out_S, 'I_predict': out_I})
x.to_csv(root + r"\MatlabCode\result.csv")
running_loss = np.append(running_loss, time_cost)
y = pd.DataFrame({'Running_loss': running_loss})
y.to_csv(root + r"\MatlabCode\loss.csv")
