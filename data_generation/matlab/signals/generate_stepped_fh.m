function [signal, params] = generate_stepped_fh(cfg)
% GENERATE_STEPPED_FH  Synthesize a single stepped frequency-hopping pulse.
%
% A stepped-frequency pulse divides T into N chips of duration Tc = T/N.
% Unlike Costas (random permutation), the chip frequencies follow a
% MONOTONIC ramp:
%
%   up   :  f_k = f_start + (k-1) * delta_f,  k = 1..N
%   down :  f_k = f_start - (k-1) * delta_f,  k = 1..N
%
% This produces a stair-step time-frequency signature that visually
% resembles a discretized LFM chirp. The model must learn to distinguish
% the smooth chirp (LFM) from the discrete steps (Stepped) — a fine
% classification challenge.
%
% Parameter ranges (set in decisions.md, 2026-05-04 Stepped/FH entry)
% ---------------------------------------------------------------
%   N             : random in {5, 6, 7, 8} (same as Costas)
%   direction     : 'up' or 'down', 50/50 random
%   delta_f       : uniform random in [2, 5] MHz (same as Costas)
%   f_start       : uniform random within band, ensuring all hops fit
%   pulse width T : uniform random in cfg.pulse_width_s (generic [1,20] us)
%
% Phase continuity
% ----------------
% As in Costas, naive switching of frequency between chips would create
% phase discontinuities and sinc-like spectral artifacts. We compute
% phase as a running integral so consecutive chips join smoothly.
%
% Inputs
% ------
%   cfg : struct from generation_config()
%
% Outputs
% -------
%   signal : (cfg.N x 1) complex double, unit-power on active region
%   params : struct
%       .class_name      = 'SteppedFH'
%       .N               = number of frequency steps (5, 6, 7, or 8)
%       .direction       = 'up' or 'down'
%       .delta_f_hz      = frequency step in Hz
%       .f_start_hz      = first chip's frequency in Hz
%       .frequencies_hz  = (1 x N) double, all hop frequencies in order
%       .pulse_width_s   = pulse duration T in seconds
%       .num_active      = pulse length in samples
%       .chip_duration_s = Tc = T / N
%       .chip_samples    = round(Tc * fs)
%       .start_idx       = index where pulse begins (1-based)
%       .stop_idx        = index where pulse ends (1-based, inclusive)

    % ------------------------------------------------------------------
    % 1) Random N (same set as Costas)
    % ------------------------------------------------------------------
    N_options = [5, 6, 7, 8];
    N = N_options(randi(numel(N_options)));

    % ------------------------------------------------------------------
    % 2) Random direction
    % ------------------------------------------------------------------
    direction = 'up';
    if rand() < 0.5
        direction = 'down';
    end

    % ------------------------------------------------------------------
    % 3) Sample frequency step delta_f in [2, 5] MHz
    % ------------------------------------------------------------------
    df_min = 2e6;
    df_max = 5e6;
    delta_f = df_min + (df_max - df_min) * rand();

    % ------------------------------------------------------------------
    % 4) Choose f_start such that ALL hops fit inside (-fs/2+m, fs/2-m)
    %
    %    For 'up'  : freqs span [f_start, f_start + (N-1)*delta_f]
    %    For 'down': freqs span [f_start - (N-1)*delta_f, f_start]
    %
    %    Total spread = (N-1)*delta_f. We choose f_start uniformly so
    %    the entire spread sits inside the band.
    % ------------------------------------------------------------------
    margin = 0.05 * cfg.fs;
    f_max_abs = cfg.fs/2 - margin;
    spread = (N - 1) * delta_f;

    if strcmp(direction, 'up')
        % f_start in [-f_max_abs, f_max_abs - spread]
        lo = -f_max_abs;
        hi =  f_max_abs - spread;
    else
        % f_start in [-f_max_abs + spread, f_max_abs]
        lo = -f_max_abs + spread;
        hi =  f_max_abs;
    end

    if hi <= lo
        % Spread too large for current band; clamp to centered placement.
        % This shouldn't happen with our settings (max spread = 7*5 MHz =
        % 35 MHz < 90 MHz available), but guard anyway.
        f_start = 0;
    else
        f_start = lo + (hi - lo) * rand();
    end

    % ------------------------------------------------------------------
    % 5) Build hop frequency sequence
    % ------------------------------------------------------------------
    if strcmp(direction, 'up')
        hop_freqs = f_start + (0:N-1) * delta_f;        % (1 x N)
    else
        hop_freqs = f_start - (0:N-1) * delta_f;        % (1 x N)
    end

    % ------------------------------------------------------------------
    % 6) Sample pulse width
    % ------------------------------------------------------------------
    pw_min = cfg.pulse_width_s(1);
    pw_max = cfg.pulse_width_s(2);
    T = pw_min + (pw_max - pw_min) * rand();

    % ------------------------------------------------------------------
    % 7) Compute chip duration & sample count
    % ------------------------------------------------------------------
    Tc = T / N;
    chip_samples = max(1, round(Tc * cfg.fs));
    num_active = chip_samples * N;
    num_active = min(num_active, cfg.N);

    % If clipping happened (rare), truncate
    full_chips_fit = floor(num_active / chip_samples);
    if full_chips_fit < N
        hop_freqs = hop_freqs(1:full_chips_fit);
        N_eff = full_chips_fit;
        num_active = chip_samples * N_eff;
    else
        N_eff = N;
    end

    % ------------------------------------------------------------------
    % 8) Build signal with continuous phase across chip boundaries
    %    (same scheme as Costas)
    % ------------------------------------------------------------------
    pulse = zeros(num_active, 1);
    phi_start = 0;     % accumulated phase at start of current chip

    for k = 1 : N_eff
        idx_start = (k - 1) * chip_samples + 1;
        idx_stop  = k * chip_samples;
        n_local = (0 : chip_samples - 1).';
        phase_k = phi_start + 2*pi * hop_freqs(k) * n_local * cfg.Ts;
        pulse(idx_start : idx_stop) = exp(1j * phase_k);
        % Increment phi_start so chip k+1 starts smoothly
        phi_start = phi_start + 2*pi * hop_freqs(k) * chip_samples * cfg.Ts;
    end

    % ------------------------------------------------------------------
    % 9) Normalize active region to unit average power
    % ------------------------------------------------------------------
    if cfg.normalize_signal_power
        p = mean(abs(pulse).^2);
        if p > 0
            pulse = pulse / sqrt(p);
        end
    end

    % ------------------------------------------------------------------
    % 10) Place pulse inside fixed-length frame
    % ------------------------------------------------------------------
    [signal, start_idx, stop_idx] = pad_signal(pulse, cfg.N, ...
                                               cfg.padding_strategy);

    % ------------------------------------------------------------------
    % 11) Pack params
    % ------------------------------------------------------------------
    params.class_name      = 'SteppedFH';
    params.N               = N_eff;
    params.direction       = direction;
    params.delta_f_hz      = delta_f;
    params.f_start_hz      = f_start;
    params.frequencies_hz  = hop_freqs;
    params.pulse_width_s   = T;
    params.num_active      = num_active;
    params.chip_duration_s = Tc;
    params.chip_samples    = chip_samples;
    params.start_idx       = start_idx;
    params.stop_idx        = stop_idx;
end
