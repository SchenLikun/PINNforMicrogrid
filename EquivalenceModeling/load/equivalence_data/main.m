clear;
clc;
%% 
Timecolumn = 1;
st = 0.01;
et = 1;

% 设置文件夹路径
folderPath = 'D:\Code\OneDrive\OneDrive - University of Edinburgh\PINNforParaEstimation\EquivalenceModeling\load\equivalence_data';  % 修改为你的 CSV 文件存储路径

% 获取文件夹中所有 CSV 文件
files = dir(fullfile(folderPath, '*.csv'));

% 初始化一个结构体数组来存储所有数据
data = struct();

for k = 1:length(files)
    % 构建完整的文件路径
    fullPath = fullfile(files(k).folder, files(k).name);

    % 读取 CSV 文件
    dataTable = readtable(fullPath);

    % 将数据存储在结构体中，以文件名为字段名
    % 去掉文件名中的非法字符（例如点和破折号）
    validName = matlab.lang.makeValidName(files(k).name);
    data.(validName) = dataTable;

    % 显示数据的前几行
    disp(['Data from ', files(k).name, ':']);
    disp(head(dataTable));
end

% 打印出所有字段名
disp('All field names in the data struct:');
disp(fieldnames(data));



BaseIndex = 1; % 选择原始文件用来做基底，其他的和这个做对比
%% normal occasion
% 
% % 设置想绘制的文件和关键词，确保文件名完全正确
% normal_Pex(1).filename = 'x0714_original_normal_csv';
% normal_Pex(1).keyword = 'P_POI1';
% normal_Pex(2).filename = 'x0714_eq_toolbox_normal_csv';
% normal_Pex(2).keyword = 'P_EQ_POI';
% normal_Pex(3).filename = 'x0714_eq_pinn_normal_csv';
% normal_Pex(3).keyword = 'P_EQ_POI';
% normal_Pex(4).filename = 'x0715_eq_pinn_normal_whole_csv';
% normal_Pex(4).keyword = 'P_EQ_POI';
% 
% 
% plotSpecificData(data, normal_Pex, Timecolumn, st , et, BaseIndex);  % 时间列索引为 1
% 
% normal_Qex(1).filename = 'x0714_original_normal_csv';
% normal_Qex(1).keyword = 'Q_POI1';
% normal_Qex(2).filename = 'x0714_eq_toolbox_normal_csv';
% normal_Qex(2).keyword = 'Q_EQ_POI';
% normal_Qex(3).filename = 'x0714_eq_pinn_normal_csv';
% normal_Qex(3).keyword = 'Q_EQ_POI';
% normal_Qex(4).filename = 'x0715_eq_pinn_normal_whole_csv';
% normal_Qex(4).keyword = 'Q_EQ_POI';
% % 
% plotSpecificData(data, normal_Qex, Timecolumn, st , et, BaseIndex);  % 时间列索引为 1

%% Area Inside 3-phase FLT
% st = 0.2;
% et = 0.35;


Inside_3phase_flt_Pex(1).filename = 'x0717_original_inner_3pflt_vf_csv';
Inside_3phase_flt_Pex(1).keyword = '101RMS';
Inside_3phase_flt_Pex(2).filename = 'x0717_toolbox_inner_3pflt_vf_csv';
Inside_3phase_flt_Pex(2).keyword = '101RMS';
Inside_3phase_flt_Pex(3).filename = 'x0717_pinn_cons_inner_3pflt_vf_csv';
Inside_3phase_flt_Pex(3).keyword = '101RMS';
Inside_3phase_flt_Pex(4).filename = 'x0717_pinn_nocons_inner_3pflt_vf_csv';
Inside_3phase_flt_Pex(4).keyword = '101RMS';
plotSpecificData(data, Inside_3phase_flt_Pex, Timecolumn, 0.2 , 0.3, BaseIndex);
plotSpecificData(data, Inside_3phase_flt_Pex, Timecolumn, st , et, BaseIndex); % 时间列索引为 1

% 
% Inside_3phase_flt_Pder(1).filename = 'x0714_original_normal_csv';
% Inside_3phase_flt_Pder(1).keyword = 'P_DG';
% Inside_3phase_flt_Pder(2).filename = 'x0714_eq_toolbox_normal_csv';
% Inside_3phase_flt_Pder(2).keyword = 'P_EQ_DG';
% Inside_3phase_flt_Pder(3).filename = 'x0714_eq_pinn_normal_csv';
% Inside_3phase_flt_Pder(3).keyword = 'P_EQ_DG';
% Inside_3phase_flt_Pder(4).filename = 'x0715_eq_pinn_normal_whole_csv';
% Inside_3phase_flt_Pder(4).keyword = 'P_EQ_DG';
% plotSpecificData(data, Inside_3phase_flt_Pder, Timecolumn, st , et, BaseIndex); % 时间列索引为 1
% plotSpecificData(data, Inside_3phase_flt_Pder, Timecolumn, 0.2 , 0.35, BaseIndex);


