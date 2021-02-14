% load smps_workspace.mat

% Skript for running a series of circuit simulations and calculating the
% corresponding harmonics via FFT
% Source "SMPS.mdl": Mahmoud Draz, DAI Labor, TU Berlin

% tests evaluation:
% - compare many cycles vs. few or only one cycle for FFT
% - compare soon after simulation start vs. late after simulation start
% - (compare different time-step sizes)

% fixed parameters
T = 1e-6;  % time-step
t = 0.2-T;  % total simulation time
f = 50;  % fundamental frequency
Va = 230;  % fundamental voltage magnitude
h_max = 350;   % highest harmonic considered

% variable operating conditions
% fh = 250;  % harmonic frequency
Initialph1 = 0.1;  % fundamental voltage phase
Initialph = 0;  % harmonic voltage phase
Vh = 2.3;  % harmonic voltage magnitude

supply_harmonics = 50*(3:2:h_max/f);
supply_voltage_h = [2.3, 23];

results = struct;

for i = (1:length(supply_harmonics))
    fh = supply_harmonics(i);
    
    % execute circuit simulation
    sim("SMPS")

    % variable evaluation parameters
    t_start = 0.06;
    cycles = 2;

    t_end = t_start + cycles/f - T;
    time_complete = (0:T:t);
    time = (t_start : T : t_end);
    L = length(time);  % time-series length
    H = (0:(L/2))/L/T;  % spectrum

    % Simulation output
    Is_complete = S_scope.signals(1).values(1:end);
    Is = S_scope.signals(1).values(int32(t_start/T+1):int32(t_end/T+1));

    % FFT
    ft = fft(Is);
    P2 = abs(ft/L);  % two-sided spectrum. abs is magnitute of frequency
    I_inj = P2(1:L/2+1);  % single sided spectrum
    % double to compensate cutting in half
    I_inj(2:end-1) = 2*I_inj(2:end-1);  

    % save paramtere and results as struct
    results(i).V_h = Vh;
    results(i).phi_v = Initialph;
    results(i).phi_f = Initialph1;
    results(i).fh = fh;
    results(i).x = H(1:int32(2*h_max/f+1));
    results(i).y = I_inj(1:int32(2*h_max/f+1));
    
    % Plot in time and frequency domain
    subplot(2,1,1)
    plot(time, Is)
    title('Current injection, time domain')
    xlabel('t (sec)')
    ylabel('Current I(t)')

    subplot(2,1,2)
    bar(H, I_inj)
    xlim([0, 1e3])
    title('Current injection, frequency domain')
    xlabel('f (Hz)')
    ylabel('Current I(f)')

end
save('circuit__sim.mat', 'results');