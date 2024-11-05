function F = objective_function(params, data)
    Pzip = params(1);
    Qzip = params(2);
    alpha_1 = params(3);
    alpha_2 = params(4);
    alpha_4 = params(5);
    alpha_5 = params(6);
    a = params(7);
    b = params(8);
    c = params(9);
    Hm = params(10);
    xr = params(11);
    xs = params(12);
    rr = params(13);
    xmu = params(14);
    rs = params(15);
    R_line = params(16);
    X_line = params(17);

    Omega = 377;  % 60Hz in rad/s
    x0 = xs + xmu;
    xp = xs + (xr * xmu) / (xr + xmu);
    T0p = (xr + xmu) / (rr * Omega);

    Id = data.Idrms;
    Iq = data.Iqrms;
    Tm = data.Tmpu;
    Vd = data.Vdrms;
    Vq = data.Vqrms;
    Wr = data.Wr;
    Te = data.Te;
    P_tot = data.P_tot;
    V_bus = data.V_bus;
    Q_tot = data.Q_tot;
    P_poi = data.P_poi;
    P_dg = data.P_dg;
    Q_poi = data.Q_poi;
    Q_dg = data.Q_dg;
    P_im = data.ph;
    Q_im = data.qh;

    alpha_3 = 1 - alpha_1 - alpha_2;
    alpha_6 = 1 - alpha_4 - alpha_5;

    % 方程 1
    eq1 = Tm - (a + b * Wr + c * Wr.^2);

    % 方程 2
    sigma_dt = - gradient(Wr, data.Time) - (Tm - Te) / (2 * Hm);

    % 方程 3
    edp = Vd + rs * Id - xp * Iq;
    eqp = Vq + rs * Iq + xp * Id;
    eq3 = - (Te * 0.15923) - (edp .* Id + eqp .* Iq);

    % 方程 4
    eq4 = edp - (Vd + (rs * Id - xp * Iq));

    % 方程 5
    eq5 = eqp - (Vq + (rs * Iq + xp * Id));

    % 方程 6
    eq6 = P_tot - Pzip * (alpha_1 * V_bus.^2 + alpha_2 * V_bus + alpha_3);

    % 方程 7
    eq7 = Q_tot - Qzip * (alpha_4 * V_bus.^2 + alpha_5 * V_bus + alpha_6);

    % 方程 8
    P_loss = P_poi + P_dg - (P_tot + P_im);
    eq8 = P_loss - ((13.8 * V_bus).^2 / (R_line^2 + (Omega * X_line)^2)) * R_line;

    % 方程 9
    Q_loss = Q_poi + Q_dg - (Q_tot + Q_im);
    eq9 = Q_loss - ((13.8 * V_bus).^2 / (R_line^2 + (Omega * X_line)^2)) * Omega * X_line;

    % 将所有方程的误差汇总
    F = [eq1; sigma_dt; eq3; eq4; eq5; eq6; eq7; eq8; eq9];
    F = sum([eq1; sigma_dt; eq3; eq4; eq5; eq6; eq7; eq8; eq9].^2);
end
