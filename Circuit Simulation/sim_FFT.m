% Script for running a series of circuit simulations and calculating the
% corresponding harmonics via FFT.

clear all

circuit_model = "SMPS";  % "EV_X" for EV_1, EV_2 and EV_4
circuit = "SMPS";

% fixed parameters
T = 1e-6;  % time-step
t = 0.2-T;  % total simulation time
h_max = 5050;   % highest frequency simulated, min = 150

% fundamental voltage source
f = 50;  % fundamental frequency
supply_voltage_f = [230*sqrt(2), 0.8*230*sqrt(2)]; %
Initialph_f_range = [0, 10];  % fundamental voltage phase, [degree]

% harmonic voltage source (variable operating conditions)
supply_harmonics = f*(3:2:h_max/f);  % harmonic frequency range
supply_voltage_h = [1.15*sqrt(2), 2.3*sqrt(2)];  % harmonic voltage magnitude range
Initialph_h = 20;  % harmonic voltage phase, [degree]

% Circuit Parameters
switch circuit
    case "SMPS"
        R1 = 0.0179;        % [ohm]
        L1 = 6e-6;          % [H], before 0.006e-6, probably mistake?
        C_emi = 35.26e-6;   % [F]
        C_dc = 0.0399;      % [F]
        R_eq = 15.11;       % [ohm]

    % the following models are from Collin.2011, Tab. II
    % collin 2011: "base power is rated power of device"
    % collin 2014: "base power is measured power draw" --> better fit
    % p_rated from collin 2014, Tab. 5.2
    case "EV_1"
        % 1 phase, bicycle charger
        p_rated = 0.11;     % [kW]
        v_dc = 315;         % [V]

        X_C_dc_pu = 0.0258;       % [pu]
        X_C_emi_pu = 9.198;       % [pu]
        X_L1_pu = 3.17e-6;        % [pu]
        R1_pu = 0.0049;         % [pu]

    case "EV_2"
        % 1 phase, moped charger
        p_rated = 0.12;      % [kW]
        v_dc = 310;         % [V]

        X_C_dc_pu = 0.0834;       % [pu]
        X_C_emi_pu = 12.58;       % [pu]
        X_L1_pu = 6.83e-5;        % [pu]
        R1_pu = 0.0028;         % [pu]

    case "EV_4"
        % 1 phase, car charger
        p_rated = 2.19;         % [kW]
        v_dc = 300;             % [V]

        X_C_dc_pu = 0.0796;       % [pu]
        X_C_emi_pu = 90.26;       % [pu]
        X_L1_pu = 6.01e-4;         % [pu]
        R1_pu = 0.0179;          % [pu]

    case "EV_5"
        % 3 phase, car charger
        p_rated = 2.18*3;         % [kW]
        v_dc = 305;             % [V]

        X_C_dc_pu = 0.447;       % [pu]
        X_C_emi_pu = 601;       % [pu]
        X_L1_pu = 7.72e-4;         % [pu]
        R1_pu = 0.0356;          % [pu]

        Initialph_f1 = 0;
        Initialph_f2 = 120;
        Initialph_f3 = 240;

        Initialph_h1 = Initialph_f1+Initialph_h;
        Initialph_h2 = Initialph_f2+Initialph_h;
        Initialph_h3 = Initialph_f3+Initialph_h;

        supply_voltage_f = supply_voltage_f/sqrt(3);
    otherwise
        disp("Circuit not available. Use SMPS, EV_1, EV_2, EV_4 or EV_5")
end

switch circuit
    case "SMPS"
        disp("No pu conversion necessary")
    case "EV_5"
        % pu System for three phases
        v_base = 230; % [V] based on v_rms

        p_base = p_rated*1000;
        i_base = p_base/v_base/sqrt(3);
        r_base = v_base/i_base/sqrt(3);

        X_C_dc = X_C_dc_pu*r_base;
        X_C_emi = X_C_emi_pu*r_base;
        X_L1 = X_L1_pu*r_base;
        R1 = R1_pu*r_base;

        R_eq = (0.006*v_dc - 0.01)*r_base;    % [ohm], from collin 2014 Eq. 5.3

        % impedance of inductors: R+jX = j omega L
        omega = 2*pi*f;
        L1 = X_L1/omega;

        % impedance of capacitors: R+jX = 1/(j omega C)
        C_dc = 1/X_C_dc/omega;
        C_emi = 1/X_C_emi/omega;
    otherwise
        % pu System for one phase
        v_base = 230; % [V] based on v_rms

        p_base = p_rated*1000;
        i_base = p_base/v_base;
        r_base = v_base/i_base;
        %l_base = r_base/(2*pi*f);

        X_C_dc = X_C_dc_pu*r_base;
        X_C_emi = X_C_emi_pu*r_base;
        X_L1 = X_L1_pu*r_base;
        R1 = R1_pu*r_base;

        R_eq = (0.006*v_dc - 0.01)*r_base;    % [ohm], from collin 2014 Eq. 5.3

        % impedance of inductors: R+jX = j omega L
        omega = 2*pi*f;
        L1 = X_L1/omega;

        % impedance of capacitors: R+jX = 1/(j omega C)
        C_dc = 1/X_C_dc/omega;
        C_emi = 1/X_C_emi/omega;
end

% variable evaluation parameters
% time of start of FFT after simulation start
t_start = 0.06;
% number of periods analyzed
% also defines interharmonic frequency resolution
cycles = 1;

% deducted simulation parameters
t_end = t_start + cycles/f - T;
time_complete = (0:T:t);
time = (t_start : T : t_end)-t_start;
L = length(time);  % time-series length
H = (0:(L/2))/L/T;  % spectrum

results_h = struct;
results_f = struct;

% simulate with only fundamental source, vary voltage phase and magnitude
for k = (1:length(Initialph_f_range))
    Initialph_f = Initialph_f_range(k);
    Va = supply_voltage_f(k);
    Vh = 0;
    fh = 0;

    % execute circuit simulation
    sim(circuit_model)

    % Simulation output
    Is_complete = S_scope.signals(1).values(1:end);
    Is = S_scope.signals(1).values(int32(t_start/T+1):int32(t_end/T+1));
    Vs_complete = S_scope.signals(2).values(1:end);
    Vs = S_scope.signals(2).values(int32(t_start/T+1):int32(t_end/T+1));

    % FFT current injections source
    ft_i = fft(Is);
    % two-sided spectrum. abs is magnitute of frequency
    P2_i = abs(ft_i/L);
    I_inj = P2_i(1:L/2+1);  % single sided spectrum
    % double to compensate cutting in half
    I_inj(2:end-1) = 2*I_inj(2:end-1);
    % phase somehow is shifted by -90 degree (bc of inductive element?)
    I_inj_phase = angle(ft_i(1:L/2+1))+pi/2;

    % FFT voltage source
    ft_v = fft(Vs);
    % two-sided spectrum. abs is magnitute of frequency
    P2_v = abs(ft_v/L);
    Vs_spec = P2_v(1:L/2+1);  % single sided spectrum
    % double to compensate cutting in half
    Vs_spec(2:end-1) = 2*Vs_spec(2:end-1);
    % phase somehow is shifted by -90 degree and interharmonics wrong
    Vs_phase = angle(ft_v(1:L/2+1))+pi/2;

    % save paramters and results of fundamental sims in struct
    results_f(k, 1).V_m_f = Va;
    results_f(k, 1).V_m_h = Vh;
    results_f(k, 1).V_a_f = Initialph_f;
    results_f(k, 1).V_a_h = Initialph_h;
    results_f(k, 1).f_h = fh;
    results_f(k, 1).H = H(1:int32(cycles*h_max/f+1));
    results_f(k, 1).I_inj = I_inj(1:int32(cycles*h_max/f+1));
    results_f(k, 1).I_inj_phase = I_inj_phase(1:int32(cycles*h_max/f+1));
    results_f(k, 1).Vs_phase = Vs_phase(1:int32(cycles*h_max/f+1));
    results_f(k, 1).t_start = t_start;
    results_f(k, 1).cycles = cycles;
    results_f(k, 1).Fs = T;
    results_f(k, 1).H_max = h_max;

    % Plot in time and frequency domain
    h_plot_max = 31;  % plot harmonics until

    subplot(2,2,1)
    plot(time*1000, Vs)
    title('Supply voltage, time domain')
    xlabel('t (ms)')
    ylabel('Voltage u (V)')

    subplot(2,2,2)
    bar(H/50, Vs_spec)
    xlim([0, h_plot_max+1])
    ylim([0, 240*sqrt(2)])
    xticks((1:2:h_plot_max))
    title('Supply voltage, frequency domain')
    xlabel('Harmonic (fund. = 50 Hz)')
    ylabel('Voltage u (V)')

    subplot(2,2,3)
    plot(time*1000, Is)
    title('Current injection, time domain')
    xlabel('t (ms)')
    ylabel('Current I (A)')

    subplot(2,2,4)
    bar(H/50, I_inj)
    xlim([0, h_plot_max+1])
    xticks((1:2:h_plot_max))
    title('Current injection, frequency domain')
    xlabel('Harmonic (fund. = 50 Hz)')
    ylabel('Current I (A)')
end

% simulate with both voltage sources, fundamental source is constant
for i = (1:length(supply_harmonics))
    for j = (1:length(supply_voltage_h))
        fh = supply_harmonics(i);
        Vh = supply_voltage_h(j);
        Initialph_f = 0;

        % execute circuit simulation
        sim(circuit_model)

        % Simulation output
        Is_complete = S_scope.signals(1).values(1:end);
        Is = S_scope.signals(1).values(int32(t_start/T+1):int32(t_end/T+1));
        Vs_complete = S_scope.signals(2).values(1:end);
        Vs = S_scope.signals(2).values(int32(t_start/T+1):int32(t_end/T+1));

        % FFT I source
        ft_i = fft(Is);
        % two-sided spectrum. abs is magnitute of frequency
        P2_i = abs(ft_i/L);
        I_inj = P2_i(1:L/2+1);  % single sided spectrum
        % double to compensate cutting in half
        I_inj(2:end-1) = 2*I_inj(2:end-1);
        % phase somehow is shifted by -90 degree (bc of inductive element?)
        I_inj_phase = angle(ft_i(1:L/2+1))+pi/2;

        % FFT V source
        ft_v = fft(Vs);
        % two-sided spectrum. abs is magnitute of frequency
        P2_v = abs(ft_v/L);
        Vs_spec = P2_v(1:L/2+1);  % single sided spectrum
        % double to compensate cutting in half
        Vs_spec(2:end-1) = 2*Vs_spec(2:end-1);
        % phase somehow is shifted by -90 degree and interharmonics wrong
        Vs_phase = angle(ft_v(1:L/2+1))+pi/2;

        % save paramters and results of harmonic sims in struct
        results_h(i, j).V_m_f = Va;
        results_h(i, j).V_m_h = Vh;
        results_h(i, j).V_a_f = Initialph_f;
        results_h(i, j).V_a_h = Initialph_h;
        results_h(i, j).f_h = fh;
        results_h(i, j).H = H(1:int32(cycles*h_max/f+1));
        results_h(i, j).I_inj = I_inj(1:int32(cycles*h_max/f+1));
        results_h(i, j).I_inj_phase = I_inj_phase(1:int32(cycles*h_max/f+1));
        results_h(i, j).Vs_phase = Vs_phase(1:int32(cycles*h_max/f+1));
        results_h(i, j).t_start = t_start;
        results_h(i, j).cycles = cycles;
        results_h(i, j).Fs = T;
        results_h(i, j).H_max = h_max;

        % Plot in time and frequency domain
        h_plot_max = 31;  % plot harmonics until

        subplot(2,2,1)
        plot(time*1000, Vs)
        title('Supply voltage, time domain')
        xlabel('t (ms)')
        ylabel('Voltage (V)')

        subplot(2,2,2)
        bar(H/50, Vs_spec)
        xlim([0, h_plot_max+1])
        ylim([0, 240*sqrt(2)])
        xticks((1:2:h_plot_max))
        title('Supply voltage, frequency domain')
        xlabel('Harmonic (fund. = 50 Hz)')
        ylabel('Voltage (V)')

        subplot(2,2,3)
        plot(time*1000, Is)
        title('Current injection, time domain')
        xlabel('t (ms)')
        ylabel('Current (A)')

        subplot(2,2,4)
        bar(H/50, I_inj)
        xlim([0, h_plot_max+1])
        xticks((1:2:h_plot_max))
        title('Current injection, frequency domain')
        xlabel('Harmonic (fund. = 50 Hz)')
        ylabel('Current (A)')
    end
end

% export results
all = struct("results_f", results_f, "results_h", results_h);
save(circuit + "_" + h_max + ".mat", 'all');
disp("Saved simulation data to " + circuit + "_" + h_max + ".mat")