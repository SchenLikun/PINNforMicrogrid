
function plotSpecificData(data, fileInfo, timeColumn, st, et, baseIndex)
    figure;  % 创建新图形
    hold on;


    % 获取基底数据
    baseData = data.(fileInfo(baseIndex).filename);
    baseTimeData = baseData.(baseData.Properties.VariableNames{timeColumn});
    
    % 计算基底数据的索引范围
    baseStartIndex = floor(st * height(baseData));
    baseEndIndex = ceil(et * height(baseData));

    baseTimeData = baseTimeData(baseStartIndex:baseEndIndex);
    baseColumn = getColumnByKeyword(baseData, fileInfo(baseIndex).keyword);
    baseColumn = baseColumn(baseStartIndex:baseEndIndex);

    for k = 1:length(fileInfo)
        if k == baseIndex
            continue;  % 跳过基底数据本身
        end

        filename = fileInfo(k).filename;
        keyword = fileInfo(k).keyword;
        dataTable = data.(filename);

        % 获取时间列数据
        timeData = dataTable.(dataTable.Properties.VariableNames{timeColumn});
        
        % 计算数据的索引范围
        startIndex = floor(st * height(dataTable));
        endIndex = ceil(et * height(dataTable));

        timeData = timeData(startIndex:endIndex);
        dataColumn = getColumnByKeyword(dataTable, keyword);
        dataColumn = dataColumn(startIndex:endIndex);

        % 计算误差
        error = dataColumn - baseColumn;
        MSE = mean((error).^2);
        MAE = mean(abs(error));
        % MRE = mean(abs(error ./ baseColumn)) * 100;

        % 打印误差信息
        disp(['MSE for ', filename, ' against base: ', num2str(MSE)]);
        disp(['MAE for ', filename, ' against base: ', num2str(MAE)]);
        % disp(['MRE for ', filename, ' against base: ', num2str(MRE), '%']);

        % 绘制误差曲线
        % plot(timeData, error, 'DisplayName', ['Error - ' filename]);
    end


    % 初始化以确定全局的最小和最大时间
    globalStart = inf;
    globalEnd = -inf;

    for k = 1:length(fileInfo)
        filename = fileInfo(k).filename;
        keyword = fileInfo(k).keyword;
        dataTable = data.(filename);

        % 获取时间列数据
        timeData = dataTable.(dataTable.Properties.VariableNames{timeColumn});

        % 获取指定的数据列
        dataColumn = getColumnByKeyword(dataTable, keyword);  % 确保这一步在循环内

        % 计算开始和结束索引
        startIndex = floor(st * height(dataTable));
        endIndex = ceil(et * height(dataTable));
        startIndex = max(1, startIndex);  % 确保不低于1
        endIndex = min(endIndex, height(dataTable));  % 确保不超出数据长度

        % 更新全局时间范围
        globalStart = min(globalStart, timeData(startIndex));
        globalEnd = max(globalEnd, timeData(endIndex));

        % 绘制数据的指定范围
        plot(timeData(startIndex:endIndex), dataColumn(startIndex:endIndex), 'DisplayName', [filename ' - ' keyword]);
    end
    
    % 设置图形属性
    title('Custom Plot for Multiple Files');
    xlabel('Time');
    ylabel('Data Value');
    legend show;
    % grid on;
    hold off;

    % 设置 X 轴范围
    xlim([globalStart, globalEnd]);
    ax = gca;
    ax.Box = 'on';
end
