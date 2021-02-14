% Skript for running a series of circuit simulations and calculating the
% corresponding harmonics via FFT
% Source: "SMPS.mdl": Mahmoud Draz, DAI Labor, TU Berlin

% tests evaluation:
% - compare many cycles vs. few or only one cycle for FFT
% - compare soon after simulation start vs. late after simulation start
% - (compare different time-step sizes)

% fixed parameters
T = 1e-6;  % time-step
t = 0.2-T;  % total simulation time
f = 50;  % fundamental frequency
Va = 230;  % fundamental voltage magnitude
h_max = 550;   % highest harmonic simulated
Initialph1 = 0.1;  % fundamental voltage phase
Initialph = 0;  % harmonic voltage phase

% variable operating conditions
% fh = 250;  % harmonic frequency
% Vh = 2.3;  % harmonic voltage magnitude

supply_harmonics = 50*(3:2:h_max/f);
supply_voltage_h = [2.3, 11.5, 23];

% variable evaluation parameters
t_start = 0.06;
cycles = 2;

results = struct;

for i = (1:length(supply_harmonics))
    for j = (1:length(supply_voltage_h))
        fh = supply_harmonics(i);
        Vh = supply_voltage_h(j);

        % execute circuit simulation
        sim("SMPS")

        t_end = t_start + cycles/f - T;
        time_complete = (0:T:t);
        time = (t_start : T : t_end);
        L = length(time);  % time-series length
        H = (0:(L/2))/L/T;  % spectrum

        % Simulation output
        Is_complete = S_scope.signals(1).values(1:end);
        Is = S_scope.signals(1).values(int32(t_start/T+1):int32(t_end/T+1));
        Vs_complete = S_scope.signals(2).values(1:end);
        Vs = S_scope.signals(2).values(int32(t_start/T+1):int32(t_end/T+1));

        % FFT
        ft_i = fft(Is);
        % two-sided spectrum. abs is magnitute of frequency
        P2_i = abs(ft_i/L);  
        I_inj = P2_i(1:L/2+1);  % single sided spectrum
        % double to compensate cutting in half
        I_inj(2:end-1) = 2*I_inj(2:end-1);
        
        ft_v = fft(Vs);
        % two-sided spectrum. abs is magnitute of frequency
        P2_v = abs(ft_v/L);  
        Vs_spec = P2_v(1:L/2+1);  % single sided spectrum
        % double to compensate cutting in half
        Vs_spec(2:end-1) = 2*Vs_spec(2:end-1);

        % save paramters and results as struct
        results(i, j).V_m_f = Va;
        results(i, j).V_m_h = Vh;
        results(i, j).V_a_f = Initialph1;
        results(i, j).V_a_h = Initialph;
        results(i, j).f_h = fh;
        results(i, j).H = H(1:int32(2*h_max/f+1));
        results(i, j).I_inj = I_inj(1:int32(2*h_max/f+1));
        results(i, j).t_start = t_start;
        results(i, j).cycles = cycles;
        results(i, j).Fs = T;
        results(i, j).H_max = h_max;

        % Plot in time and frequency domain
        subplot(2,2,1)
        plot(time, Vs)
        title('Supply voltage, time domain')
        xlabel('t (sec)')
        ylabel('Voltage V(t)')
        
        subplot(2,2,2)
        bar(H, Vs_spec)
        xlim([0, 1e3])
        title('Supply voltage, frequency domain')
        xlabel('f (Hz)')
        ylabel('Voltage V(f)')

        subplot(2,2,3)
        plot(time, Is)
        title('Current injection, time domain')
        xlabel('t (sec)')
        ylabel('Current I(t)')

        subplot(2,2,4)
        bar(H, I_inj)
        xlim([0, 1e3])
        title('Current injection, frequency domain')
        xlabel('f (Hz)')
        ylabel('Current I(f)')
    end
end

% export results
save('circuit_sim.mat', 'results');