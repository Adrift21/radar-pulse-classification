function [signal, params] = generate_costas(cfg)
% GENERATE_COSTAS  Synthesize a single Costas frequency-hopping pulse.
%
% A Costas-coded pulse divides T into N chips of duration Tc = T/N. In
% each chip k = 1..N, the carrier transmits at a hop frequency
%
%   f_k = (pi(k) - (N+1)/2) * delta_f
%
% where pi is a Costas permutation of {1, ..., N}. The "(N+1)/2" offset
% centers the hop set around 0 Hz (symmetric baseband). The Costas
% property (distinct difference vectors) yields a thumbtack ambiguity
% function with very low side-lobes — the reason these codes are used.
%
% Parameter ranges (set in decisions.md, 2026-05-04 Costas entry)
% ---------------------------------------------------------------
%   N             : random in {5, 6, 7, 8}, equal probability
%   sequence pi   : two predefined sequences per N, picked at random
%   delta_f       : uniform random in [2, 5] MHz
%   pulse width T : uniform random in cfg.pulse_width_s (generic [1,20] us)
%
% Phase continuity
% ----------------
% Naive switching of frequency between chips introduces phase
% discontinuities and hence sinc-like spectral artifacts on each hop.
% We compute the phase as a running integral so that consecutive chips
% join smoothly:
%
%   phi(t) = phi(t_{k-1}) + 2*pi * f_k * (t - t_{k-1})  for t in chip k
%
% Inputs
% ------
%   cfg : struct from generation_config()
%
% Outputs
% -------
%   signal : (cfg.N x 1) complex double, unit-power on active region
%   params : struct
%       .class_name      = 'Costas'
%       .N               = code length (5, 6, 7, or 8)
%       .sequence        = (1 x N) int8, the Costas permutation pi(k)
%       .delta_f_hz      = frequency step in Hz
%       .frequencies_hz  = (1 x N) double, actual hop frequencies
%       .pulse_width_s   = pulse duration T in seconds
%       .num_active      = pulse length in samples
%       .chip_duration_s = Tc = T / N
%       .chip_samples    = round(Tc * fs)
%       .start_idx       = index where pulse begins (1-based)
%       .stop_idx        = index where pulse ends (1-based, inclusive)

    % ------------------------------------------------------------------
    % 1) Costas sequence library (2 sequences per N, equal probability)
    % ------------------------------------------------------------------
    costas_lib = struct();
    costas_lib.N5 = { [3,1,4,2,5], ...
                      [2,4,1,5,3] };
    costas_lib.N6 = { [4,1,6,3,5,2], ...
                      [5,2,1,3,6,4] };
    costas_lib.N7 = { [4,7,1,6,5,2,3], ...
                      [3,2,5,7,4,1,6] };
    costas_lib.N8 = { [3,5,8,7,2,1,4,6], ...
                      [5,7,2,8,3,1,4,6] };

    % ------------------------------------------------------------------
    % 2) Random N, then random sequence within that N
    % ------------------------------------------------------------------
    N_options = [5, 6, 7, 8];
    N = N_options(randi(numel(N_options)));
    seq_set = costas_lib.(sprintf('N%d', N));
    pi_seq = int8(seq_set{randi(numel(seq_set))});      % (1 x N)

    % ------------------------------------------------------------------
    % 3) Sample frequency step delta_f in [2, 5] MHz
    % ------------------------------------------------------------------
    df_min = 2e6;
    df_max = 5e6;
    delta_f = df_min + (df_max - df_min) * rand();

    % Hop frequencies, centered around 0 Hz (symmetric baseband)
    % f_k = (pi(k) - (N+1)/2) * delta_f
    hop_freqs = (double(pi_seq) - (N + 1) / 2) * delta_f;     % (1 x N)

    % Safety: ensure all hop frequencies stay within (-fs/2 + margin)
    margin = 0.05 * cfg.fs;
    f_max_abs = cfg.fs/2 - margin;
    if max(abs(hop_freqs)) > f_max_abs
        % Should not occur with our delta_f range and N <= 8, but guard
        scale = f_max_abs / max(abs(hop_freqs));
        delta_f = delta_f * scale;
        hop_freqs = hop_freqs * scale;
    end

    % ------------------------------------------------------------------
    % 4) Sample pulse width (generic range)
    % ------------------------------------------------------------------
    pw_min = cfg.pulse_width_s(1);
    pw_max = cfg.pulse_width_s(2);
    T = pw_min + (pw_max - pw_min) * rand();

    % ------------------------------------------------------------------
    % 5) Compute chip duration & sample count
    % ------------------------------------------------------------------
    Tc = T / N;
    chip_samples = max(1, round(Tc * cfg.fs));
    num_active = chip_samples * N;
    num_active = min(num_active, cfg.N);

    % If clipping happened (rare), truncate
    full_chips_fit = floor(num_active / chip_samples);
    if full_chips_fit < N
        pi_seq = pi_seq(1:full_chips_fit);
        hop_freqs = hop_freqs(1:full_chips_fit);
        N_eff = full_chips_fit;
        num_active = chip_samples * N_eff;
    else
        N_eff = N;
    end

    % ------------------------------------------------------------------
    % 6) Build signal with continuous phase across chip boundaries
    %
    %    For chip k (samples [(k-1)*chip_samples+1 .. k*chip_samples]),
    %    use phase:
    %      phi(n) = phi_chip_start_k + 2*pi * f_k * (n - n_start_k) * Ts
    %    where phi_chip_start_k is set so that the phase is continuous
    %    with the end of chip k-1.
    % ------------------------------------------------------------------
    pulse = zeros(num_active, 1);
    phi_start = 0;     % accumulated phase at start of current chip

    for k = 1 : N_eff
        idx_start = (k - 1) * chip_samples + 1;
        idx_stop  = k * chip_samples;
        % local time within chip, samples 0..chip_samples-1
        n_local = (0 : chip_samples - 1).';
        phase_k = phi_start + 2*pi * hop_freqs(k) * n_local * cfg.Ts;
        pulse(idx_start : idx_stop) = exp(1j * phase_k);
        % Accumulate phase up to the end-of-chip (the *next* sample
        % continues from chip_samples * Ts, hence the use of
        % chip_samples in the increment so chip k+1 starts smoothly)
        phi_start = phi_start + 2*pi * hop_freqs(k) * chip_samples * cfg.Ts;
    end

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
    params.class_name      = 'Costas';
    params.N               = N_eff;
    params.sequence        = pi_seq;
    params.delta_f_hz      = delta_f;
    params.frequencies_hz  = hop_freqs;
    params.pulse_width_s   = T;
    params.num_active      = num_active;
    params.chip_duration_s = Tc;
    params.chip_samples    = chip_samples;
    params.start_idx       = start_idx;
    params.stop_idx        = stop_idx;
end
