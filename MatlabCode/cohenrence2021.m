% 用于NCS的相干解调计算
% created by：YSL
% date：2023.6
function [N, C, S] = cohenrence2021(Y, cos_delta_2,cos_delta_1Minus2,cos_delta_1Plus2,sin_delta_1Minus2,sin_delta_1Plus2, window, sample_points)
% 用于NCS的FFT计算
% N, C, S：薄膜参数，列向量（N×1）
% Y：测量光谱，列向量（N×1）
% cos_delta2, cos_delta_1Minus2, sin_delta_1Minus2：channel仪器的波片延迟量
%% 默认参数设置
% 设置窗函数
if window==0
    win = ones(sample_points, 1);
else
    eval(strcat('win = ', window, "(sample_points);"));
end
%% 使用傅里叶方法提取Stokes参量
% 画出测量光谱的OPD域图并选频
% 提取N
f = fftshift(fft(Y.*cos_delta_2.*win));
figure;
plot(abs(f), 'LineWidth', 1.6);
title('please choose the 0 channel')
set(gca, 'Fontsize', 16, 'Fontname', 'Times New Roman')
[f_cat, ~] = ginput(2);
ch0 = zeros(sample_points,1);
for i = round(f_cat(1)):round(f_cat(2))
    ch0(i) = f(i);
end
N = -1*real(ifft(ifftshift(ch0))./win);
close
% 提取C
% 提取C1
f1 = fftshift(fft(Y.*cos_delta_1Minus2.*win));
figure;
plot(abs(f1), 'LineWidth', 1.6);
title('please choose the 0 channel')
set(gca, 'Fontsize', 16, 'Fontname', 'Times New Roman')
[f_cat, ~] = ginput(2);
ch1 = zeros(sample_points,1);
for i = round(f_cat(1)):round(f_cat(2))
    ch1(i) = f1(i);
end
C1 = 2*real(ifft(ifftshift(ch1))./win);
close
% 提取C2
f1 = fftshift(fft(Y.*cos_delta_1Plus2.*win));
figure;
plot(abs(f1), 'LineWidth', 1.6);
title('please choose the 0 channel')
set(gca, 'Fontsize', 16, 'Fontname', 'Times New Roman')
[f_cat, ~] = ginput(2);
ch1 = zeros(sample_points,1);
for i = round(f_cat(1)):round(f_cat(2))
    ch1(i) = f1(i);
end
C2 = -2*real(ifft(ifftshift(ch1))./win);
close
C = (C1 + C2) / 2;
% 提取S
% 提取S1
f2 = fftshift(fft(Y.*sin_delta_1Minus2.*win));
figure;
plot(abs(f2), 'LineWidth', 1.6);
title('please choose the 0 channel')
set(gca, 'Fontsize', 16, 'Fontname', 'Times New Roman')
[f_cat, ~] = ginput(2);
ch1 = zeros(sample_points,1);
for i = round(f_cat(1)):round(f_cat(2))
    ch1(i) = f2(i);
end
S1 = 2*real(ifft(ifftshift(ch1))./win);
close
% 提取S2
f2 = fftshift(fft(Y.*sin_delta_1Plus2.*win));
figure;
plot(abs(f2), 'LineWidth', 1.6);
title('please choose the 0 channel')
set(gca, 'Fontsize', 16, 'Fontname', 'Times New Roman')
[f_cat, ~] = ginput(2);
ch1 = zeros(sample_points,1);
for i = round(f_cat(1)):round(f_cat(2))
    ch1(i) = f2(i);
end
S2 = -2*real(ifft(ifftshift(ch1))./win);
close
S = (S1 + S2) / 2;
end