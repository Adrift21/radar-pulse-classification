function [signal, params] = generate_lfm(cfg)
% GENERATE_LFM  Synthesize a single LFM (Linear FM / chirp) pulse.
%
% Generates a complex baseband LFM signal of random pulse width and
% bandwidth, embedded in a fixed-length frame of cfg.N samples. The pulse
% is placed within the frame according to cfg.padding_strategy and
% normalized to unit average power on its active region.
%
% Inputs
% ------
%   cfg : struct from generation_config()
%
% Outputs
% -------
%   signal : (cfg.N x 1) complex double, unit-power on active region
%   params : struct with the parameters actually used (for logging/debug)
%       .class_name      = 'LFM'
%       .pulse_width_s   = pulse duration in seconds
%       .num_active      = pulse length in samples
%       .bandwidth_hz    = chirp bandwidth in Hz
%       .f_start_hz      = chirp start frequency in Hz
%       .f_stop_hz       = chirp stop frequency in Hz
%       .chirp_rate_hzps = k = B/T in Hz/s
%       .direction       = 'up' or 'down'
%       .start_idx       = index where pulse begins (1-based)
%       .stop_idx        = index where pulse ends (1-based, inclusive)
%
% Notes
% -----
% - Working entirely in complex baseband. The "carrier" here is just the
%   chirp center frequency offset from DC; classes like CW will use a
%   single carrier instead.
% - We constrain f_start and f_stop to lie within (-fs/2, fs/2) to avoid
%   aliasing. A safety margin of 5% of fs is enforced.

    % ------------------------------------------------------------------
    % 1) Sample pulse width
    % ------------------------------------------------------------------
    pw_min = cfg.pulse_width_s(1);
    pw_max = cfg.pulse_width_s(2);
    T = pw_min + (pw_max - pw_min) * rand();          % uniform in [min,max]
    num_active = round(T * cfg.fs);                   % samples in pulse
    num_active = min(num_active, cfg.N);              % safety clip

    % ------------------------------------------------------------------
    % 2) Sample chirp bandwidth and direction
    % ------------------------------------------------------------------
    B_min = cfg.lfm_bandwidth_hz(1);
    B_max = cfg.lfm_bandwidth_hz(2);
    B = B_min + (B_max - B_min) * rand();             % bandwidth [Hz]

    direction = 'up';
    if rand() < 0.5
        direction = 'down';
    end

    % ------------------------------------------------------------------
    % 3) Choose chirp center frequency, derive f_start / f_stop
    %    Constrain to (-fs/2 + margin, fs/2 - margin).
    % ------------------------------------------------------------------
    margin = 0.05 * cfg.fs;                           % 5% guard band
    f_max_abs = cfg.fs/2 - margin - B/2;              % center-freq bound
    if f_max_abs <= 0
        % Bandwidth too large for current fs; fallback to centered chirp
        fc = 0;
    else
        fc = -f_max_abs + 2 * f_max_abs * rand();     % uniform in band
    end

    if strcmp(direction, 'up')
        f_start = fc - B/2;
        f_stop  = fc + B/2;
    else
        f_start = fc + B/2;
        f_stop  = fc - B/2;
    end

    k = (f_stop - f_start) / T;                       % chirp rate [Hz/s]

    % ------------------------------------------------------------------
    % 4) Synthesize the chirp on its active time grid
    % ------------------------------------------------------------------
    t = (0 : num_active - 1).' * cfg.Ts;              % column vector
    phase = 2 * pi * (f_start .* t + 0.5 * k .* t.^2);
    pulse = exp(1j * phase);                          % complex baseband

    % ------------------------------------------------------------------
    % 5) Normalize active region to unit average power
    % ------------------------------------------------------------------
    if cfg.normalize_signal_power
        p = mean(abs(pulse).^2);
        if p > 0
            pulse = pulse / sqrt(p);
        end
    end

    % ------------------------------------------------------------------
    % 6) Place pulse inside fixed-length frame using padding strategy
    % ------------------------------------------------------------------
    [signal, start_idx, stop_idx] = pad_signal(pulse, cfg.N, ...
                                               cfg.padding_strategy);

    % ------------------------------------------------------------------
    % 7) Pack params for logging
    % ------------------------------------------------------------------
    params.class_name      = 'LFM';
    params.pulse_width_s   = T;
    params.num_active      = num_active;
    params.bandwidth_hz    = B;
    params.f_start_hz      = f_start;
    params.f_stop_hz       = f_stop;
    params.chirp_rate_hzps = k;
    params.direction       = direction;
    params.start_idx       = start_idx;
    params.stop_idx        = stop_idx;
end
