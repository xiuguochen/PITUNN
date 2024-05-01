% 按选定采样点数读取csv测量数据
% created by：YSL
% date：2023.4
function y = read_data(filename, sample_points, wvn)
% 按选定采样点数读取txt测量数据
% y：读出的光谱数据，行向量
% filename：测量数据文件名，字符串
% sample_points：采样点数，正整数（默认400）
% wvn：采样点波数的起始点和终止点，数组（默认[1/0.8, 1/0.4]）
%% 默认参数设置
if nargin==1
    sample_points = 400;
    wvn = [1/0.8, 1/0.4];
elseif nargin==2
    wvn = [1/0.8, 1/0.4];
end
%% 读取数据
opts = detectImportOptions(filename,'VariableNamingRule', "preserve"); % 根据基于文件内容生成导入选项
Tdata(:,1:2) = readtable(filename,opts); % 波数和光强数据导入至table
Tdata.Properties.VariableNames(1:2) = {'wavenumber','inten'}; % 修改光强数据的变量名称

% 换算变量单位
Tdata.wavenumber = 1./(Tdata.wavenumber*1e-3); % 波长转变为波数，单位：μm-1

% 插值到采样点处的数据
wvn_begin = wvn(1); wvn_last = wvn(2);
wvn_set = wvn_begin:(wvn_last-wvn_begin)/(sample_points-1):wvn_last;
y = spline(Tdata.wavenumber,Tdata.inten,wvn_set);
end