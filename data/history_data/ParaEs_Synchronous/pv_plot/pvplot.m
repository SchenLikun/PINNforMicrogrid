% Load data
input_data = readtable('data/1011PVtest.csv');
fnamevar = "variables_testpv1014.dat";

% Rename and drop columns
input_data.Properties.VariableNames = {'Time', 'V', 'I', 'Is'};
input_data.Is = [];

% Calculate derivatives
step_time = input_data.Time(2) - input_data.Time(1);
dI_dt = (input_data.I(3:end) - input_data.I(1:end-2)) / step_time / 2;
dV_dt = (input_data.V(3:end) - input_data.V(1:end-2)) / step_time / 2;

% Read the output data in fnamevar
fid = fopen(fnamevar, 'r');
lines = textscan(fid, '%s', 'Delimiter', '\n');
lines = lines{1};
fclose(fid);

Chat = [];
for i = 1:length(lines)
    match = regexp(lines{i}, '\[(.*?)\]', 'tokens');
    if ~isempty(match)
        line_data = str2double(strsplit(match{1}{1}, ','));
        Chat = [Chat; line_data];
    end
end

% Extract variables
Kt = Chat(:, 1);
Rs = Chat(:, 2);
Rsh = Chat(:, 3) * 1e4;
Iso = Chat(:, 4);

V_mat = repmat(input_data.V(2:end-1), 1, length(Kt));
I_mat = repmat(input_data.I(2:end-1), 1, length(Kt));
dI_dt_mat = repmat(dI_dt, 1, length(Kt));
dV_dt_mat = repmat(dV_dt, 1, length(Kt));

Rs_mat = repmat(Rs', size(I_mat, 1), 1);
Kt_mat = repmat(Kt', size(I_mat, 1), 1);
Rsh_mat = repmat(Rsh', size(I_mat, 1), 1);

left = 1 + (Iso' .* Rs' ./ Kt') .* exp((V_mat + I_mat .* Rs_mat) ./ Kt_mat) + Rs_mat ./ Rsh_mat;
right = -dV_dt_mat .* ((Iso' ./ Kt_mat) .* exp((V_mat + I_mat .* Rs_mat) ./ Kt_mat) + 1 ./ Rsh_mat);
result = left .* dI_dt

% Plotting results
figure;
plot(mean(abs(result), 1));
title('Mean of Absolute Value of Result');

figure;
plot(mean(abs(result), 1));
hold on;
plot(1:size(Chat, 1), ones(1, size(Chat, 1)) * 0, "k--");
ylim([-0.1, 1]);
title('Mean of Absolute Value of Result with Horizontal Line');

figure;
plot(mean(abs(result), 1));
hold on;
plot(1:size(Chat, 1), ones(1, size(Chat, 1)) * 0, "k--");
xlim([0, 1000]);
title('Zoomed-in View of Previous Plot');
