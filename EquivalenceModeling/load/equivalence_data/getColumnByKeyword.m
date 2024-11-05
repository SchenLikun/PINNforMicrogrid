function dataColumn = getColumnByKeyword(dataTable, keyword)
    % 搜索与关键字匹配的列名
    colNames = dataTable.Properties.VariableNames;
    matchedColIdx = find(contains(colNames, keyword, 'IgnoreCase', true));
    if isempty(matchedColIdx)
        error('No columns match the provided keyword.');
    elseif length(matchedColIdx) > 1
        error('Multiple columns match the keyword. Please refine your search.');
    end
    % 返回匹配的列数据
    dataColumn = dataTable.(colNames{matchedColIdx});
end