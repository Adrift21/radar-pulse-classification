function [signal, params] = generate_cw(cfg)
% GENERATE_CW  Synthesize a single continuous-wave (CW) pulse.
%
% A CW pulse is the simplest waveform in the dataset: a single tone
% at a fixed carrier frequency, with no modulation throughout the
% pulse. Used by simple radars and as an electronic-warfare baseline.
%
% Signal model
% ------------
%   s(t) = exp( j * (2*pi * f_c * t + phi_0) )
% where:
%   - f_c   : carrier frequency (uniform random over the available band)
%   - phi_0 : initial phase (uniform random in [0, 2*pi))
%
% Inputs
% ------
%   cfg : struct from generation_config()
%
% Outputs
% -------
%   signal : (cfg.N x 1) complex double, unit-power on active region
%   params : struct
%       .class_name      = 'CW'
%       .f_carrier_hz    = carrier frequency in Hz
%       .initial_phase   = initial phase phi_0 in radians
%       .pulse_width_s   = pulse duration T in seconds
%       .num_active      = pulse length in samples
%       .start_idx       = index where pulse begins (1-based)
%       .stop_idx        = index where pulse ends (1-based, inclusive)
%
% Notes
% -----
% - f_c is constrained to (-fs/2 + margin, fs/2 - margin) with the same
%   5%-of-fs guard band used by LFM. With fs = 100 MHz this gives
%   approximately +/- 45 MHz.
% - Active-region power is exactly 1 by construction (|exp(j*phase)| = 1),
%   so the optional normalization step is a no-op but kept for consistency
%   with other generators.

    % ------------------------------------------------------------------
    % 1) Sample pulse width from the generic range
    % ------------------------------------------------------------------
    pw_min = cfg.pulse_width_s(1);
    pw_max = cfg.pulse_width_s(2);
    T = pw_min + (pw_max - pw_min) * rand();
    num_active = round(T * cfg.fs);
    num_active = min(num_active, cfg.N);

    % ------------------------------------------------------------------
    % 2) Sample carrier frequency f_c with guard band
    % ------------------------------------------------------------------
    margin = 0.05 * cfg.fs;
    f_max_abs = cfg.fs/2 - margin;
    fc = -f_max_abs + 2 * f_max_abs * rand();

    % ------------------------------------------------------------------
    % 3) Sample initial phase phi_0
    % ------------------------------------------------------------------
    phi0 = 2 * pi * rand();

    % ------------------------------------------------------------------
    % 4) Synthesize the tone on its active time grid
    % ------------------------------------------------------------------
    t = (0 : num_active - 1).' * cfg.Ts;
    pulse = exp(1j * (2*pi * fc .* t + phi0));

    % ------------------------------------------------------------------
    % 5) Normalize active region to unit average power
    %    (already unit power for pure tone, kept for API consistency)
    % ------------------------------------------------------------------
    if cfg.normalize_signal_power
        p = mean(abs(pulse).^2);
        if p > 0
            pulse = pulse / sqrt(p);
        end
    end

    % ------------------------------------------------------------------
    % 6) Place pulse inside fixed-length frame
    % ------------------------------------------------------------------
    [signal, start_idx, stop_idx] = pad_signal(pulse, cfg.N, ...
                                               cfg.padding_strategy);

    % ------------------------------------------------------------------
    % 7) Pack params
    % ------------------------------------------------------------------
    params.class_name    = 'CW';
    params.f_carrier_hz  = fc;
    params.initial_phase = phi0;
    params.pulse_width_s = T;
    params.num_active    = num_active;
    params.start_idx     = start_idx;
    params.stop_idx      = stop_idx;
end
