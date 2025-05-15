function [reward_value, transmission_info] = simulate_transmission(arm, config)
    % Extract parameters
    SF = arm(1); Tx = arm(2); BW = arm(3);
    V = config.V; d = config.d;
    
    % Lookup tables
    I_table = [-18, 6.2; -12, 6.8; -3, 8.8; 0, 10.1; 3, 12.0; 9, 17.7; 12.5, 24.0];
    BW_list = [203 406 812 1625];
    sensitivity_table = [
        -109 -111 -115 -118 -121 -124 -127 -130;
        -107 -110 -113 -116 -119 -122 -125 -128;
        -105 -108 -112 -115 -117 -120 -123 -126;
        -99  -103 -106 -109 -111 -114 -117 -120
    ];
    
    % Calculate parameters
    I = interp1(I_table(:,1), I_table(:,2), Tx, 'linear', 'extrap');
    T_sym = (2^SF) / (BW * 1e3);
    
    % Calculate number of symbols
    if SF < 7
        N_sym = 8 + 6.25 + 8 + ceil(max(8*config.N_payload + config.CRC - 4*SF + config.H, 0)/(4*SF)) * (config.CR+4);
    elseif SF <= 10
        N_sym = 8 + 4.25 + 8 + ceil(max(8*config.N_payload + config.CRC - 4*SF + 8 + config.H, 0)/(4*SF)) * (config.CR+4);
    else
        N_sym = 8 + 4.25 + 8 + ceil(max(8*config.N_payload + config.CRC - 4*SF + 8 + config.H, 0)/(4*(SF-2))) * (config.CR+4);
    end
    
    ToA = T_sym * N_sym;
    
    % Calculate power levels
    Lp = 40 + 50 * log10(d) + 6 + 3;
    G = 2; L = 2;
    P_rx = Tx + G - Lp - L + G - L;
    sf_interp = interp1([5 6 7 8 9 10 11 12], 1:8, SF);
    bw_idx = find(BW_list == BW);
    P_rx_min = interp1(1:8, sensitivity_table(bw_idx,:), sf_interp);
    
    % Simulate transmission
    success = false;
    for attempts = 1:8
        noise = -0.5 + rand();
        if P_rx + noise > P_rx_min
            success = true;
            break;
        end
    end
    
    % Calculate energy and cost
    E_transmit = I * V * ToA;
    cost = E_transmit * attempts + config.penalty * (~success);
    reward_value = 1 - cost / config.cost_max;
    
    transmission_info = struct('E_transmit', E_transmit, 'cost', cost, ...
        'attempts', attempts, 'ToA', ToA, 'I', I, 'P_rx', P_rx, ...
        'P_rx_min', P_rx_min);
end