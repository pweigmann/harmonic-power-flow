% Skript for running a series of circuit simulations and calculating the
% corresponding harmonics via FFT. 
% Source: "SMPS.mdl": Mahmoud Draz, DAI Labor, TU Berlin

% tests evaluation:
% - compare many cycles vs. few or only one cycle for FFT
% - compare soon after simulation start vs. late after simulation start
% - (compare different time-step sizes)
% - compare NE for different operating conditions
    
clear all

circuit = "SMPS";

% fixed parameters
T = 1e-6;  % time-step
t = 0.2-T;  % total simulation time
h_max = 500;   % highest frequency simulated, min = 150

% fundamental voltage source
f = 50;  % fundamental frequency
supply_voltage_f = [230, 200]; %
Va = 230;  % fundamental voltage magnitude
Initialph_f_range = [0, 10];  % fundamental voltage phase, [degree]

% harmonic voltage source (variable operating conditions)
supply_harmonics = 50*(3:2:h_max/f);  % harmonic frequency range
supply_voltage_h = [2.3, 23];  % harmonic voltage magnitude range
Initialph_h = 20;  % harmonic voltage phase, [degree]

% variable evaluation parameters
% time of start of FFT after simulation start
t_start = 0.06;  
% number of periods analyzed
% also defines interharmonic frequency resolution
cycles = 2;

% deducted simulation parameters
t_end = t_start + cycles/f - T;
time_complete = (0:T:t);
time = (t_start : T : t_end);
L = length(time);  % time-series length
H = (0:(L/2))/L/T;  % spectrum

results_f = struct;
results_h = struct;

% simulate with only fundamental source, vary voltage phase and magnitude
for k = (1:length(Initialph_f_range))
    Initialph_f = Initialph_f_range(k);
    Va = supply_voltage_f(k);
    Vh = 0;
    fh = 0;
    
    % execute circuit simulation
    sim(circuit)

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
    h_plot_max = 21;  % plot harmonics until

    subplot(2,2,1)
    plot(time, Vs)
    title('Supply voltage, time domain')
    xlabel('t (sec)')
    ylabel('Voltage V(t)')

    subplot(2,2,2)
    bar(H/50, Vs_spec)
    xlim([0, h_plot_max+1])
    ylim([0, 240])
    xticks((1:2:h_plot_max))
    title('Supply voltage, frequency domain')
    xlabel('Harmonic (fund. = 50 Hz)')
    ylabel('Voltage V(f)')

    subplot(2,2,3)
    plot(time, Is)
    title('Current injection, time domain')
    xlabel('t (sec)')
    ylabel('Current I(t)')

    subplot(2,2,4)
    bar(H/50, I_inj)
    xlim([0, h_plot_max+1])
    xticks((1:2:h_plot_max))
    title('Current injection, frequency domain')
    xlabel('Harmonic (fund. = 50 Hz)')
    ylabel('Current I(f)')
end

% simulate with both voltage sources, fundamental source is constant
for i = (1:length(supply_harmonics))
    for j = (1:length(supply_voltage_h))
        fh = supply_harmonics(i);
        Vh = supply_voltage_h(j);
        Initialph_f = 0;

        % execute circuit simulation
        sim("SMPS")

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
        h_plot_max = 21;  % plot harmonics until
        
        subplot(2,2,1)
        plot(time, Vs)
        title('Supply voltage, time domain')
        xlabel('t (sec)')
        ylabel('Voltage (V)')
        
        subplot(2,2,2)
        bar(H/50, Vs_spec)
        xlim([0, h_plot_max+1])
        ylim([0, 240])
        xticks((1:2:h_plot_max))
        title('Supply voltage, frequency domain')
        xlabel('Harmonic (fund. = 50 Hz)')
        ylabel('Voltage (V)')

        subplot(2,2,3)
        plot(time, Is)
        title('Current injection, time domain')
        xlabel('t (sec)')
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
%save(circuit + ".mat", 'all');
