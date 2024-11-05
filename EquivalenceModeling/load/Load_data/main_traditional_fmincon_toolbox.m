clear;
clc;

%% 

% 读取CSV文件，并保留原始列标题
data = readtable('Load_data/0616_island_IM.csv', 'VariableNamingRule', 'preserve');

% 字典：原始列名到新列名的映射
dict_col = containers.Map({'Time', 'I107d_rms', 'I107q_rms', 'Omega107V', 'Tmpu_M1', 'V107d_rms', 'V107q_rms', ...
    'PSTAT1M1', 'QSTAT1M1', 'SPDOUT1M1', 'TELECT1M1', 'I1070_rms', 'V1070_rms', 'P_POI1', ...
    'Ptot_Area1', 'P_DG', 'Q_POI1', 'Qtot_Area1', 'Q_DG', 'V101RMS'}, ...
    {'Time', 'Idrms', 'Iqrms', 'Ws', 'Tmpu', 'Vdrms', 'Vqrms', 'ph', 'qh', 'Wr', 'Te', 'I0rms', 'V0rms', ...
    'P_poi', 'P_tot', 'P_dg', 'Q_poi', 'Q_tot', 'Q_dg', 'V_bus'});

% 提取有用的列名部分
original_column_names = data.Properties.VariableNames;
extracted_column_names = cellfun(@(x) regexp(x, '[^|]+$', 'match', 'once'), original_column_names, 'UniformOutput', false);

% 重命名表头
new_column_names = cellfun(@(x) dict_col(x), extracted_column_names, 'UniformOutput', false);
data.Properties.VariableNames = new_column_names;

% 提取需要的数据列
time = data.Time;
Tm = data.Tmpu;
Wr = data.Wr;
Te = data.Te;
Id = data.Idrms;
Iq = data.Iqrms;
Vd = data.Vdrms;
Vq = data.Vqrms;
P_tot = data.P_tot;
V_bus = data.V_bus;
Q_tot = data.Q_tot;
P_poi = data.P_poi;
P_dg = data.P_dg;
Q_poi = data.Q_poi;
Q_dg = data.Q_dg;
P_im = data.ph;
Q_im = data.qh;

%% 
% 定义初始变量值
initial_params = [
    1;    % Pzip
    1;    % Qzip
    1;    % alpha_1
    1;    % alpha_2
    1;    % alpha_4
    1;    % alpha_5
    1;    % a
    1;    % b
    1;    % c
    1;    % Hm
    1; % xr
    1;% xs
    1; % rr
    1;    % xmu
    1; % rs
    1;    % R_line
    1;    % X_line
];
%% 

% 使用 fmincon 优化求解，添加约束
options = optimoptions('fmincon', 'Display', 'iter', 'MaxIterations', 10000, 'MaxFunctionEvaluations', 100000);
[param_estimates, resnorm] = fmincon(@(params)objective_function(params, data), initial_params, [], [], [], [], [], [], @(params)constraint_function(params), options);


% 使用 patternsearch 优化求解，不添加约束
% options = optimoptions('patternsearch', 'Display', 'iter', 'MaxIterations', 10000, 'MaxFunctionEvaluations', 100000);
% [param_estimates, resnorm] = patternsearch(@(params)objective_function(params, data), initial_params, [], [], [], [], [], [], options);


% 输出估计的参数值
disp('估计的参数值：');
disp(param_estimates);
% P_model = Pzip * (alpha_1 * V_bus.^2 + alpha_2 * V_bus + alpha_3);
% Q_model = Qzip * (alpha_4 * V_bus.^2 + alpha_5 * V_bus + alpha_6);
% % 可视化结果
% figure;
% plot(time, P_tot, 'bo');  % 
% hold on;
% plot(time, P_model, 'r--');  %
% xlabel('Time');  % 
% ylabel('P_{tot}');
% legend('Actual Data', 'Model Prediction');
% title('Model Output vs Actual Data - P_{tot}');
% 
% figure;
% plot(time, Q_tot, 'bo');  % 
% hold on;
% plot(time, Q_model, 'r--');  %
% xlabel('Time');  % 
% ylabel('Q_{tot}');
% legend('Actual Data', 'Model Prediction');
% title('Model Output vs Actual Data - Q_{tot}');
% 
% % 其他变量的拟合结果可视化
% Tm_model = param_estimates(7) + param_estimates(8) * Wr + param_estimates(9) * Wr.^2;
% figure;
% plot(time, Tm, 'bo'); 
% hold on;
% plot(time, Tm_model, 'r--');  
% xlabel('Time');  
% ylabel('Tm');
% legend('Actual Data', 'Model Prediction');
% title('Model Output vs Actual Data - Tm');
% 
% % 计算并可视化 sigma_dt
% sigma_dt_model = - gradient(Wr, time) - (Tm_model - Te) / (2 * param_estimates(10));  % 修改此处
% figure;
% plot(time, sigma_dt_model, 'r--'); 
% xlabel('Time'); 
% ylabel('sigma\_dt');
% title('Model Prediction - sigma\_dt');



