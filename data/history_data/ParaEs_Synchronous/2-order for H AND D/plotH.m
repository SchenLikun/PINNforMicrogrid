% clear the data
clear;
close all;

% read file path
file_read = dir('D:\Code\OneDrive\OneDrive - University of Edinburgh\PINNforParaEstimation\ParaEs_Synchronous\2-order for H AND D\*.csv');

% read file name and len
filename = {file_read.name};
file_length = length(file_read);

% read the csv file data
CsvData = [];

for i=1:file_length
    id = file_read(i).name;
    path = strcat('D:\Code\OneDrive\OneDrive - University of Edinburgh\PINNforParaEstimation\ParaEs_Synchronous\2-order for H AND D\','\',id);
    file_data = csvread(id);
    CsvData = [CsvData, file_data(:,2), file_data(:,3)];
end
% get the size of dataframe
[m, n] = size(CsvData);

for i = 1 : m
    for j = 1 : n
        Hmin = min(min(CsvData(i,1),CsvData(i,3)),CsvData(i,5));
        Hmax = max(max(CsvData(i,1),CsvData(i,3)),CsvData(i,5));
        % 7 8 列为H的包络线,135为H代表性训练结果
        CsvData(i,7) = Hmin;
        CsvData(i,8) = Hmax;
        Dmin = min(min(CsvData(i,2),CsvData(i,4)),CsvData(i,6));
        Dmax = max(max(CsvData(i,2),CsvData(i,4)),CsvData(i,6));
        % 7 8 列为D的包络线,246为D代表性训练结果
        CsvData(i,9) = Dmin;
        CsvData(i,10) = Dmax;
    end
end

save('D:\Code\OneDrive\OneDrive - University of Edinburgh\PINNforParaEstimation\ParaEs_Synchronous\2-order for H AND D\csvdata.mat','CsvData');

for k = 1:2:5
    plot(CsvData(:,k));
    hold on;
end

area_plot = [];
for l = 7:10
    area_plot = [area_plot, CsvData(:,l)];
end
%% 绘制包络线

x=1:length(area_plot(:,1)); %横坐标
H1=area_plot(:,2); %H下曲线
H2=area_plot(:,1); %H上曲线
D1=area_plot(:,4); %D下曲线
D2=area_plot(:,3); %D上曲线
%绘图
H_area = [H1; flip(H2)];
D_area = [D1; flip(D2)];
pic01 = fill([x,fliplr(x)],H_area,'r');
% pic02 = fill([x,fliplr(x)],D_area,'b');
%改变边缘和阴影透明度

set(pic01,'edgealpha', 0, 'facealpha', 0.4);
% set(pic02,'edgealpha', 0, 'facealpha', 0.4);


