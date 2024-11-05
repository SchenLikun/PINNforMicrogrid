function plotDataFromFiles(dataStruct, keyword, timeColumn)
    % 创建新图形
    figure;
    hold on;

    % 提取结构体中的所有字段名
    fileNames = fieldnames(dataStruct);

    % 遍历每个文件的数据
    for k = 1:length(fileNames)
        dataTable = dataStruct.(fileNames{k});

        % 获取时间列数据
        timeData = dataTable.(dataTable.Properties.VariableNames{timeColumn});

        % 使用关键字获取数据列
        try
            dataColumn = getColumnByKeyword(dataTable, keyword);
            plot(timeData, dataColumn, 'DisplayName', fileNames{k});
        catch ME
            disp(['Error in file ', fileNames{k}, ': ', ME.message]);
        end
    end

    % 设置图形属性
    title(['Plot for keyword: ', keyword]);
    xlabel('Time');
    ylabel('Data Value');
    legend show;
    grid on;
    hold off;
end