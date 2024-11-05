function [c, ceq] = constraint_function(params)
    rs = params(15);
    xs = params(12);
    xmu = params(14);
    rr = params(13);
    xr = params(11);
    Hm = params(10);

    % 定义不等式约束
    c = [
        2e-2 - rs;   % rs >= 2e-2
        rs - 5e-2;   % rs <= 5e-2
        1e-1 - xs;   % xs >= 1e-1
        xs - 5e-1;   % xs <= 5e-1
        0.75 - xmu;  % xmu >= 0.75
        xmu - 10;    % xmu <= 10
        2e-2 - rr;   % rr >= 2e-2
        rr - 5e-2;   % rr <= 5e-2
        1e-1 - xr;   % xr >= 1e-1
        xr - 5e-1;   % xr <= 5e-1
        0.05 - Hm;   % Hm >= 0.05
        Hm - 1       % Hm <= 1
    ];
    % 定义等式约束
    ceq = [];
end
