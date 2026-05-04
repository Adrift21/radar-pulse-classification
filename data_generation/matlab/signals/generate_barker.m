function [signal, params] = generate_barker(cfg)
% GENERATE_BARKER  Synthesize a single Barker-coded BPSK pulse.
%
% Generates a complex baseband Barker phase-coded signal. One of three
% canonical Barker codes is selected at random with equal probability:
%
%   B7  : [+1 +1 +1 -1 -1 +1 -1]
%   B11 : [+1 +1 +1 -1 -1 -1 +1 -1 -1 +1 -1]
%   B13 : [+1 +1 +1 +1 +1 -1 -1 +1 +1 -1 +1 -1 +1]
%
% Signal model
% ------------
%   s(t) = c(t) * exp(j*2*pi*fc*t)
% where c(t) is a piecewise-constant {+1,-1} chip sequence of length N
% with chip duration Tc = T/N. Phase changes are instantaneous (no
% pulse-shaping filter); see decisions.md (2026-05-04 Barker entry)
% for the rationale.
%
% Inputs
% ------
%   cfg : struct from generation_config()
%
% Outputs
% -------
%   signal : (cfg.N x 1) complex double, unit-power on active region
%   params : struct
%       .class_name      = 'Barker'
%       .code_name       = 'B7' | 'B11' | 'B13'
%       .code_length     = 7 | 11 | 13
%       .code_sequence   = (1xN) int8 in {+1,-1}
%       .pulse_width_s   = total pulse duration T in seconds
%       .num_active      = pulse length in samples
%       .chip_duration_s = Tc = T / code_length
%       .chip_samples    = round(Tc * fs)
%       .f_carrier_hz    = carrier frequency fc in Hz
%       .start_idx       = index where pulse begins (1-based)
%       .stop_idx        = index where pulse ends (1-based, inclusive)
%
% Notes
% -----
% - The chip-sample mapping uses round(); residual fractional samples
%   are absorbed into the last chip, so num_active is exactly
%   chip_samples * code_length, which may differ from round(T*fs) by
%   up to a few samples. This is normal and within the variability
%   already present from the random pulse-width draw.
% - We require chip_samples >= 1; with cfg.fs=100 MHz and
%   T_min = 1 us, the worst case is B13 -> Tc = 1us/13 ~ 77 ns ->
%   ~7.7 samples, well above 1.

    % ------------------------------------------------------------------
    % 1) Select Barker code at random (equal probability)
    % ------------------------------------------------------------------
    barker_codes = struct();
    barker_codes.B7  = int8([ 1  1  1 -1 -1  1 -1]);
    barker_codes.B11 = int8([ 1  1  1 -1 -1 -1  1 -1 -1  1 -1]);
    barker_codes.B13 = int8([ 1  1  1  1  1 -1 -1  1  1 -1  1 -1  1]);

    code_names = {'B7', 'B11', 'B13'};
    pick = code_names{randi(3)};
    code_seq = barker_codes.(pick);
    Ncode = numel(code_seq);

    % ------------------------------------------------------------------
    % 2) Sample pulse width
    % ------------------------------------------------------------------
    pw_min = cfg.pulse_width_s(1);
    pw_max = cfg.pulse_width_s(2);
    T = pw_min + (pw_max - pw_min) * rand();

    % ------------------------------------------------------------------
    % 3) Compute chip duration & sample count
    % ------------------------------------------------------------------
    Tc = T / Ncode;                          % chip duration [s]
    chip_samples = max(1, round(Tc * cfg.fs)); % samples per chip
    num_active = chip_samples * Ncode;        % total active samples
    num_active = min(num_active, cfg.N);     % safety clip

    % If clipping happened, we may not get full code length; recompute
    % how many full chips actually fit.
    full_chips_fit = floor(num_active / chip_samples);
    if full_chips_fit < Ncode
        % Truncate code if frame too short (very rare with our settings)
        code_seq = code_seq(1:full_chips_fit);
        Ncode = full_chips_fit;
        num_active = chip_samples * Ncode;
    end

    % ------------------------------------------------------------------
    % 4) Choose carrier frequency (same range as LFM/NLFM, with margin)
    % ------------------------------------------------------------------
    margin = 0.05 * cfg.fs;
    f_max_abs = cfg.fs/2 - margin;
    fc = -f_max_abs + 2 * f_max_abs * rand();

    % ------------------------------------------------------------------
    % 5) Build chip envelope c(t) by repeating each code element
    %    chip_samples times.
    % ------------------------------------------------------------------
    % repelem replicates each element of code_seq chip_samples times.
    % Result is a (num_active x 1) column vector of {+1,-1}.
    c = double(repelem(code_seq(:), chip_samples));

    % ------------------------------------------------------------------
    % 6) Modulate onto carrier (complex baseband)
    % ------------------------------------------------------------------
    t = (0 : num_active - 1).' * cfg.Ts;
    pulse = c .* exp(1j * 2*pi * fc .* t);

    % ------------------------------------------------------------------
    % 7) Normalize active region to unit average power
    % ------------------------------------------------------------------
    if cfg.normalize_signal_power
        p = mean(abs(pulse).^2);
        if p > 0
            pulse = pulse / sqrt(p);
        end
    end

    % ------------------------------------------------------------------
    % 8) Place pulse inside fixed-length frame
    % ------------------------------------------------------------------
    [signal, start_idx, stop_idx] = pad_signal(pulse, cfg.N, ...
                                               cfg.padding_strategy);

    % ------------------------------------------------------------------
    % 9) Pack params
    % ------------------------------------------------------------------
    params.class_name      = 'Barker';
    params.code_name       = pick;
    params.code_length     = Ncode;
    params.code_sequence   = code_seq;
    params.pulse_width_s   = T;
    params.num_active      = num_active;
    params.chip_duration_s = Tc;
    params.chip_samples    = chip_samples;
    params.f_carrier_hz    = fc;
    params.start_idx       = start_idx;
    params.stop_idx        = stop_idx;
end
