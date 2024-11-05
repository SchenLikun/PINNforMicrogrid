% 加载数据并保留原始列标题
% filename = '0616_flt_IM.csv';
% filename = '0711_outflt_IM.csv';
filename = '0616_island_IM.csv';
filename = '0616_normal_IM.csv';
opts = detectImportOptions(filename, 'VariableNamingRule', 'preserve');
data = readtable(filename, opts);

% 重命名列
data.Properties.VariableNames = {'Time', 'I0rms', 'Idrms', 'Iqrms', 'Ws', 'P_poi','P_tot1', 'Q_poi', 'Q_tot1', 'Tmpu', 'V101bus', 'V0rms', 'Vdrms', 'Vqrms', 'p_dg','q_dg', 'ph', 'qh', 'Wr', 'Te'};

% 计算 phCalc
v1 = @(row) (row.Vdrms * row.Idrms + row.Vqrms * row.Iqrms );
data.phCalc = arrayfun(@(idx) v1(data(idx,:)), (1:height(data))');

% 绘制 ph 和 phCalc
figure;
plot(data.Time, data.ph, data.Time, data.phCalc);
legend('ph', 'phCalc');

% 描述 phCalc 和 ph 的差异
ph_diff = data.phCalc - data.ph;
disp('phCalc 和 ph 的差异:');
disp(mean(ph_diff));
% disp(mean(ph_diff./data.ph)*100);

% 计算 qhCalc
v2 = @(row) (row.Vqrms * row.Idrms - row.Vdrms * row.Iqrms);
data.qhCalc = arrayfun(@(idx) v2(data(idx,:)), (1:height(data))');

% 绘制 qh 和 qhCalc
figure;
plot(data.Time, data.qh, data.Time, data.qhCalc);
legend('qh', 'qhCalc');

% 描述 qhCalc 和 qh 的差异
qh_diff = data.qhCalc - data.qh;
disp('qhCalc 和 qh 的差异:');
disp(mean(qh_diff));
% disp(mean(qh_diff./data.qh)*100);

% 定义参数
rs =  2.0110e-2;
xs =  1.0448e-1;
xmu = 9.0424;
rr = 4.5768e-2;
xr = 1.0448e-1;
Omega_b = 377; % 60Hz
Hm = 0.075352957;
Te_base = 0.15923;

x0 = xs + xmu;
xp = xs + xr * xmu / (xr + xmu);
T0p = (xr + xmu) / (Omega_b * rr);

% 计算 edp 和 eqp
data.edp = data.Vdrms - (rs * data.Idrms - xp * data.Iqrms);
data.eqp = data.Vqrms - (rs * data.Iqrms + xp * data.Idrms);

% 计算 TeCalc
data.TeCalc = (data.edp .* data.Idrms + data.eqp .* data.Iqrms);
data.TeReal = - data.Te * Te_base; % p.u. to real value

% 绘制 TeCalc 和 TeReal
figure;
plot(data.Time, data.TeCalc, data.Time, data.TeReal);
legend('TeCalc', 'TeReal');

% 描述 TeCalc 和 TeReal 的差异
Te_diff = data.TeCalc - data.TeReal;
disp('TeCalc 和 TeReal 的差异:');
disp(mean(Te_diff));
% disp(mean(Te_diff./data.Te)*100);

% 计算滑差 sigma 和 sigma_dot
data.sigma = 1 - data.Wr;
sigma_dt = [diff(data.sigma) ./ diff(data.Time); NaN];  % 用 NaN 填充尾部

% 确保 sigma_dt 的长度与 data 的高度互相匹配
if length(sigma_dt) ~= height(data)
    error('sigma_dt 的长度与原始数据高度不匹配');
end
data.sigma_dot = sigma_dt;

% 计算 sigma_Calc
data.sigma_Calc = (data.Te - data.Tmpu) / (2 * Hm);

% 绘制 sigma_dot 和 sigma_Calc
figure;
plot(data.Time, data.sigma_dot, data.Time, data.sigma_Calc);
legend('sigma_dot', 'sigma_Calc');

% 描述 sigma_dot 和 sigma_Calc 的差异
sigma_diff = data.sigma_dot - data.sigma_Calc;
disp('sigma_dot 和 sigma_Calc 的差异:');
disp(mean(sigma_diff, 'omitnan'));
sigma_percentage_diff = (sigma_diff ./ data.sigma_Calc) * 100;
disp('sigma_diff 的百分比差异的平均值:');
disp(mean(sigma_percentage_diff, 'omitnan'));

% 计算数值导数
edp_dt = [diff(data.edp) ./ diff(data.Time); NaN];  % 用 NaN 填充尾部
eqp_dt = [diff(data.eqp) ./ diff(data.Time); NaN];  % 用 NaN 填充尾部

% 确保 edp_dt 和 eqp_dt 的长度与 data 的高度匹配
if length(edp_dt) ~= height(data) || length(eqp_dt) ~= height(data)
    error('数值导数的长度与原始数据高度不匹配');
end
data.edp_dot_data = edp_dt;
data.eqp_dot_data = eqp_dt;

% 计算 edp_dot_Calc 和 eqp_dot_Calc
data.edp_dot_Calc = Omega_b * data.sigma .* data.eqp - (data.edp + (x0 - xp) * data.Iqrms) / T0p;
data.eqp_dot_Calc = - Omega_b * data.sigma .* data.edp - (data.eqp - (x0 - xp) * data.Idrms) / T0p;
data.eqp_dot_Modify = data.eqp_dot_Calc + 2.192;

% 绘制 eqp_dot_data 和 eqp_dot_Modify
figure;
plot(data.Time, data.eqp_dot_data, data.Time, data.eqp_dot_Modify);
legend('eqp_dot_data', 'eqp_dot_Modify');

% 描述 eqp_dot_data 和 eqp_dot_Modify 的差异
eqp_diff = data.eqp_dot_Modify - data.eqp_dot_data;
disp('eqp_dot_Modify 和 eqp_dot_data 的差异:');
disp(mean(eqp_diff, 'omitnan'));



figure;
plot(data.Time, data.edp_dot_data, data.Time, data.edp_dot_Calc);
legend('edp_dot_data', 'edp_dot_Calc');

% 验证计算
data.edp_dot_Calc1 = Omega_b * data.sigma .* data.eqp - (data.edp + (x0 - xp) * data.Iqrms) / T0p;
data.edp_dot_Calc2 = - (data.edp + (x0 - xp) * data.Iqrms) / T0p;
data.edp_dot_Calc3 = Omega_b * data.sigma .* data.eqp;
T0p_Verify = - 1 ./ ((data.edp_dot_data - Omega_b * data.sigma .* data.eqp) ./ (data.edp + (x0 - xp) * data.Iqrms));

% 描述 T0p_Verify
disp('T0p_Verify:');
disp(mean(T0p_Verify, 'omitnan'));
