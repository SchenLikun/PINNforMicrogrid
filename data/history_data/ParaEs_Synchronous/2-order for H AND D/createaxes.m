function createaxes(Parent1, YMatrix1)
%CREATEAXES(Parent1, YMatrix1)
%  PARENT1:  axes parent
%  YMATRIX1:  plot y 数据的矩阵

%  由 MATLAB 于 18-Oct-2023 17:56:17 自动生成

% 创建 axes
axes1 = axes('Parent',Parent1);
hold(axes1,'on');

% 使用 plot 的矩阵输入创建多个 line 对象
plot1 = plot(YMatrix1,'Parent',axes1);
set(plot1(1),'DisplayName','CsvData(:,2)');
set(plot1(2),'DisplayName','CsvData(:,3)');
set(plot1(3),'DisplayName','CsvData(:,5)');
set(plot1(4),'DisplayName','CsvData(:,6)');
set(plot1(5),'DisplayName','CsvData(:,8)');
set(plot1(6),'DisplayName','CsvData(:,9)');
set(area);

box(axes1,'on');
hold(axes1,'off');
% 设置其余坐标区属性
set(axes1,'ContextMenu');
