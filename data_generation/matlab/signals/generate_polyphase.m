function [signal, params] = generate_polyphase(cfg)
% GENERATE_POLYPHASE  Synthesize a single P1/P2/P3/P4 polyphase pulse.
%
% One of four Lewis-Kretschmer polyphase codes is selected at random
% with equal probability. All four codes use a base size N drawn from
% {4, 6, 8} (same set as Frank), with N_c = N^2 total chips.
%
% Phase formulas
% --------------
%
%   P1:  phi(m,n) = -(pi/N) * (N - (2n-1)) * ((n-1)*N + (m-1))
%        Read row-major from N x N matrix, m,n = 1..N.
%
%   P2:  phi(m,n) = -(pi / (2N)) * (2m - 1 - N) * (2n - 1 - N)
%        Read row-major from N x N matrix, m,n = 1..N.
%        Requires N even (all our N values 4,6,8 are even, OK).
%
%   P3:  phi(k) = pi * (k-1)^2 / N_c,  k = 1..N_c
%        Direct 1-D index, no matrix.
%
%   P4:  phi(k) = pi * (k-1)^2 / N_c - pi * (k-1),  k = 1..N_c
%        Same as P3 plus a linear term (palindromic frequency profile).
%
% Signal model (zero carrier, complex baseband)
% ---------------------------------------------
%   s(t) = exp(j * phi_chip(k)),    k = floor(t / Tc) + 1
%
% No additional carrier modulation: f_c = 0 (consistent with Frank).
% See decisions.md (2026-05-04 P1-P4 entry) for rationale.
%
% Inputs
% ------
%   cfg : struct from generation_config()
%
% Outputs
% -------
%   signal : (cfg.N x 1) complex double, unit-power on active region
%   params : struct
%       .class_name      = 'Polyphase'
%       .subcode         = 'P1' | 'P2' | 'P3' | 'P4'
%       .N               = matrix base size (4, 6, or 8)
%       .num_chips       = N_c = N^2
%       .phase_sequence  = (1 x N_c) double, phases in radians
%       .phase_matrix    = (N x N) double for P1/P2; [] for P3/P4
%       .pulse_width_s   = total pulse duration T in seconds
%       .num_active      = pulse length in samples
%       .chip_duration_s = Tc = T / N_c
%       .chip_samples    = round(Tc * fs)
%       .start_idx       = index where pulse begins (1-based)
%       .stop_idx        = index where pulse ends (1-based, inclusive)
%
% Notes
% -----
% - Uses cfg.polyphase_pulse_width_s (shared with Frank).
% - Like Barker and Frank, chip transitions are instantaneous; no
%   pulse-shaping filter is applied.

    % ------------------------------------------------------------------
    % 1) Select sub-code at random (equal probability)
    % ------------------------------------------------------------------
    subcodes = {'P1', 'P2', 'P3', 'P4'};
    subcode = subcodes{randi(numel(subcodes))};

    % ------------------------------------------------------------------
    % 2) Select matrix size N (same set as Frank)
    % ------------------------------------------------------------------
    N_options = [4, 6, 8];
    N = N_options(randi(numel(N_options)));
    num_chips = N * N;

    % ------------------------------------------------------------------
    % 3) Build phase sequence according to selected sub-code
    % ------------------------------------------------------------------
    phase_matrix = [];   % only filled for matrix-based codes (P1, P2)

    switch subcode
        case 'P1'
            % phi(m,n) = -(pi/N) * (N - (2n-1)) * ((n-1)*N + (m-1))
            phase_matrix = zeros(N, N);
            for m = 1:N
                for n = 1:N
                    phase_matrix(m, n) = -(pi / N) * (N - (2*n - 1)) ...
                                         * ((n - 1) * N + (m - 1));
                end
            end
            phase_sequence = reshape(phase_matrix.', 1, []);

        case 'P2'
            % phi(m,n) = -(pi / (2N)) * (2m - 1 - N) * (2n - 1 - N)
            phase_matrix = zeros(N, N);
            for m = 1:N
                for n = 1:N
                    phase_matrix(m, n) = -(pi / (2*N)) ...
                                         * (2*m - 1 - N) ...
                                         * (2*n - 1 - N);
                end
            end
            phase_sequence = reshape(phase_matrix.', 1, []);

        case 'P3'
            % phi(k) = pi * (k-1)^2 / N_c,  k = 1..N_c
            k_idx = 0 : (num_chips - 1);
            phase_sequence = pi * (k_idx.^2) / num_chips;

        case 'P4'
            % phi(k) = pi * (k-1)^2 / N_c - pi * (k-1)
            k_idx = 0 : (num_chips - 1);
            phase_sequence = pi * (k_idx.^2) / num_chips - pi * k_idx;

        otherwise
            error('generate_polyphase:badSubcode', ...
                  'Unknown subcode: %s', subcode);
    end

    % Wrap phases to [-pi, pi) for cleaner numerical behavior
    phase_sequence = mod(phase_sequence + pi, 2*pi) - pi;

    % ------------------------------------------------------------------
    % 4) Sample pulse width from shared polyphase range
    % ------------------------------------------------------------------
    pw_min = cfg.polyphase_pulse_width_s(1);
    pw_max = cfg.polyphase_pulse_width_s(2);
    T = pw_min + (pw_max - pw_min) * rand();

    % ------------------------------------------------------------------
    % 5) Compute chip duration & sample count
    % ------------------------------------------------------------------
    Tc = T / num_chips;
    chip_samples = max(1, round(Tc * cfg.fs));
    num_active = chip_samples * num_chips;
    num_active = min(num_active, cfg.N);

    % If clipping happened, truncate code
    full_chips_fit = floor(num_active / chip_samples);
    if full_chips_fit < num_chips
        phase_sequence = phase_sequence(1:full_chips_fit);
        num_chips = full_chips_fit;
        num_active = chip_samples * num_chips;
    end

    % ------------------------------------------------------------------
    % 6) Expand phase sequence to per-sample phase array
    % ------------------------------------------------------------------
    phase_per_sample = repelem(phase_sequence(:), chip_samples);

    % ------------------------------------------------------------------
    % 7) Build complex baseband signal (no carrier)
    % ------------------------------------------------------------------
    pulse = exp(1j * phase_per_sample);

    % ------------------------------------------------------------------
    % 8) Normalize active region to unit average power
    % ------------------------------------------------------------------
    if cfg.normalize_signal_power
        p = mean(abs(pulse).^2);
        if p > 0
            pulse = pulse / sqrt(p);
        end
    end

    % ------------------------------------------------------------------
    % 9) Place pulse inside fixed-length frame
    % ------------------------------------------------------------------
    [signal, start_idx, stop_idx] = pad_signal(pulse, cfg.N, ...
                                               cfg.padding_strategy);

    % ------------------------------------------------------------------
    % 10) Pack params
    % ------------------------------------------------------------------
    params.class_name      = 'Polyphase';
    params.subcode         = subcode;
    params.N               = N;
    params.num_chips       = num_chips;
    params.phase_sequence  = phase_sequence;
    params.phase_matrix    = phase_matrix;   % [] for P3/P4
    params.pulse_width_s   = T;
    params.num_active      = num_active;
    params.chip_duration_s = Tc;
    params.chip_samples    = chip_samples;
    params.start_idx       = start_idx;
    params.stop_idx        = stop_idx;
end
