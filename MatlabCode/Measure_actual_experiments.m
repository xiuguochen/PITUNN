% 用于实验结果的处理
% created by：YSL
% date：2023.6
%% 初始变量设置
clc
clear
close all
addpath(genpath(pwd))
root = ' C:\Users\98072\ysl_file\PITUNN_code'; % 运行前修改根目录
sample_points = 400;
wvn = [1/0.8, 1/0.4]; % 采样点波数的起始点和终止点
match_range = 0.1*sample_points:0.9*sample_points; % 评价用波数范围
window_name = 'blackman';
wvn_begin = wvn(1); wvn_last = wvn(2);
wavenumber = (wvn_begin:(wvn_last-wvn_begin)/(sample_points-1):wvn_last)';
 % 测量数据储存位置 
filename_measure = "20230719_Al-Al2O3_20.37nm.csv";
hyper_paras1 = [0.06; 0.02];
hyper_paras2 = [0.03; 0.08; 0.03; 0.03; 0.08; 0.3];
data = readtable(filename_measure);
Iout = data.Iout_exp; Iin = data.Iin_fit;
cos_delta_2 = real(data.e_delta2);
cos_delta_1Minus2 = real(data.e_delta1_delta2); sin_delta_1Minus2 = imag(data.e_delta1_delta2);
cos_delta_1Plus2 = real(data.e_delta1plusdelta2); sin_delta_1Plus2 = imag(data.e_delta1plusdelta2);
cos_delta_1 = real(data.e_delta1); sin_delta_1 = imag(data.e_delta1);
N_RC2 = data.N_RC2; C_RC2 = data.C_RC2; S_RC2 = data.S_RC2;
%% 画图展示测量光谱和真实值
% A = zero_clean(Iout, Iin, window_name, sample_points);
% Y = 8*Iout./Iin./A;
spectrum = load('Al-Al2O3.mat');
Y = spectrum.Y;
% figure;
% plot(wavenumber, Y, 'Color', [0 0 0]);
% xlabel('wavenumber (\mum^{-1})')
% ylabel('inten')
% xlim(wvn)
% set(gca, 'Fontsize', 16, 'Fontname', 'Times New Roman')
% figure;
% plot(wavenumber,N_RC2,wavenumber,C_RC2,wavenumber,S_RC2);
% xlabel('wavenumber (\mum^{-1})')
% xlim(wvn)
% legend('N-RC2','C-RC2','S-RC2')
% set(gca, 'Fontsize', 16, 'Fontname', 'Times New Roman')
%% CSE -- FFT/coh
% 傅里叶变换/相干解调提取技术
tic
[N_coh, C_coh, S_coh] = FFT_extract(Iin', Iout', wavenumber',...
    data.e_delta2', data.e_delta1_delta2', data.e_delta1plusdelta2',...
    data.e_delta1', sample_points');
toc
%% CSE -- compressed sensing
tic
[N_compressed,C_compressed,S_compressed,M_basis] = compressed_sensing...
    (Y,cos_delta_2,cos_delta_1Minus2,sin_delta_1Minus2,cos_delta_1Plus2,...
    sin_delta_1Plus2,5,hyper_paras1(1),hyper_paras1(2),sample_points,'DCT+L');
toc
%% CSE -- DIP-SP
T1 = table(hyper_paras1);
writetable(T1,"hyper.csv");
T2 = table(Y,cos_delta_2,cos_delta_1Minus2,sin_delta_1Minus2,cos_delta_1Plus2,sin_delta_1Plus2,cos_delta_1,sin_delta_1,...
    'VariableNames',{'Y','cos_delta_2','cos_delta_1Minus2','sin_delta_1Minus2','cos_delta_1Plus2','sin_delta_1Plus2','cos_delta_1','sin_delta_1'});
writetable(T2,"data.csv");
T3 = table(M_basis);
writetable(T3,"M_basis.csv");
command = 'activate';
system(command,"-echo")
command1_1 = strcat('python', root);
command1 = strcat(command1_1, '\finalPythonProject\main_DIPSP.py');
system(command1,"-echo")
% 读取结果
T_result = readtable("result.csv");
T_loss = readtable("loss.csv");
N_DIP = T_result.out_N;
C_DIP = T_result.out_C;
S_DIP = T_result.out_S;
loss_DIP = T_loss.Running_loss(1:399);
%% CSE -- PTUNN
% 读取数据并储存传递给python代码
% T2 = table(Y,cos_delta_2,cos_delta_1Minus2,sin_delta_1Minus2,cos_delta_1Plus2,sin_delta_1Plus2,cos_delta_1,sin_delta_1,...
%     'VariableNames',{'Y','cos_delta_2','cos_delta_1Minus2','sin_delta_1Minus2','cos_delta_1Plus2','sin_delta_1Plus2','cos_delta_1','sin_delta_1'});
% writetable(T2,"data.csv");
% T3 = table(M_basis);
% writetable(T3,"M_basis.csv");
T4 = table(hyper_paras2);
writetable(T4,"hyper.csv");
command = 'activate';
system(command,"-echo")
command1_1 = strcat('python', root);
command1 = strcat(command1_1, '\finalPythonProject\main.py');
system(command1,"-echo")
% 读取结果
T_result = readtable("result.csv");
T_loss = readtable("loss.csv");
N_net = T_result.out_N;
C_net = T_result.out_C;
S_net = T_result.out_S;

%% 方法效果对比
% show_box(wavenumber,match_range,N_DIP,N_compressed,N_net,N_RC2,C_DIP,C_compressed,C_net,C_RC2,S_DIP,S_compressed,S_net,S_RC2)
% Ipre = T_result.I_predict;figure; plot(wavenumber, Y, wavenumber,Ipre); figure; plot(N_net.^2+C_net.^2+S_net.^2);
[~,~,~,r_coh] = RMSE(N_coh,N_RC2,C_coh,C_RC2,-S_coh,S_RC2,match_range);
[~,~,~,r_compressed] = RMSE(N_compressed,N_RC2,C_compressed,C_RC2,-S_compressed,S_RC2,match_range);
[~,~,~,r_DIP] = RMSE(N_DIP,N_RC2,C_DIP,C_RC2,-S_DIP,S_RC2,match_range);
[~,~,~,r_net] = RMSE(N_net,N_RC2,C_net,C_RC2,-S_net,S_RC2,match_range);
disp(r_coh);
disp(r_compressed);disp(r_DIP);disp(r_net)
%%
createfigure(wavenumber,N_coh,N_RC2,C_coh,C_RC2,-S_coh,S_RC2,match_range,...
    false, [-0.01,0.05], true)
createfigure(wavenumber,N_compressed,N_RC2,C_compressed,C_RC2,-S_compressed,...
    S_RC2,match_range,false, [-0.01,0.05], true)
createfigure(wavenumber,N_DIP,N_RC2,C_DIP,C_RC2,-S_DIP,S_RC2,match_range,...
    false, [-0.01,0.05], true)
createfigure(wavenumber,N_net,N_RC2,C_net,C_RC2,-S_net,S_RC2,match_range,...
    false, [-0.01,0.05], true)



