% 清除工作区、关闭图形窗口和清除命令窗口
close all;
clear;
clc;

% 读取CSV数据文件
filename = '927testforTe2.csv';
data = readmatrix(filename, 'HeaderLines', 1);

% 提取第一列数据作为横坐标
x = data(:, 1);

% 提取第二到四列数据作为纵坐标
y = data(:, 2:4);

% 采样数据，每隔n个点采样一次
n=180;
sampledIndices = 1:n:size(y, 1);
sampledY = y(sampledIndices, :);

% 定义采样点的颜色
sampledColors = ['#10729e'; '#f2931e'; '#ae4132'];  %十六进制颜色代码

% 绘制数据图表
plot(x, y);
hold on; % 保持图形，以便添加采样数据和额外的点
plot(x(sampledIndices), sampledY, 'o');

% 使用不同颜色的实心点标记列2到列4的采样点
for i = 1:3
    plot(x(sampledIndices), sampledY(:, i), 'o', 'MarkerFaceColor', sampledColors(i, :));
end

% 添加坐标轴标签和标题
xlabel('Time');
ylabel('Value(p.u.)');
% title('数据图表');

% 添加图例，说明哪些是采样点和列2到列4的数据
% legend('Tm', 'w', 'Te', '列2采样点', '列3采样点', '列4采样点');

hold off; % 完成图形
