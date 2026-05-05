% TEST_NLFM_VISUALIZATION
% Sanity check for NLFM generator. Produces:
%   - Figure 1: 6 NLFM examples (mix of quadratic + sinusoidal variants)
%              Shows real part, instantaneous frequency, STFT spectrogram.
%   - Figure 2: Side-by-side comparison of LFM vs NLFM-quadratic vs
%              NLFM-sinusoidal at the same SNR for visual discrimination.
%   - Figure 3: NLFM at multiple SNRs (does signal survive at -10 dB?).
%
% Run from data_generation/matlab/ directory:
%   >> cd data_generation/matlab
%   >> addpath(genpath(pwd))
%   >> test_nlfm_visualization

clear; clc; close all;

addpath('config', 'signals', 'utils');

cfg = generation_config();

% ------------------------------------------------------------------
% Figure 1: 6 NLFM pulses (clean), inspecting variant mix
% ------------------------------------------------------------------
figure('Name', 'NLFM clean pulses', 'Position', [60 60 1300 800]);

variant_count = struct('quadratic', 0, 'sinusoidal', 0);

for k = 1:6
    [s, p] = generate_nlfm(cfg);
    variant_count.(p.variant) = variant_count.(p.variant) + 1;

    t = (0:cfg.N-1) * cfg.Ts * 1e6;        % us

    subplot(6, 3, (k-1)*3 + 1);
    plot(t, real(s), 'b'); hold on; plot(t, imag(s), 'r');
    xlabel('t [\mus]'); ylabel('amp');
    title(sprintf('NLFM #%d  [%s]  T=%.2f \\mus  B=%.1f MHz', ...
          k, p.variant, p.pulse_width_s*1e6, p.bandwidth_hz/1e6));
    legend('I','Q', 'Location', 'eastoutside'); grid on;

    subplot(6, 3, (k-1)*3 + 2);
    inst_phase = unwrap(angle(s));
    inst_freq  = [0; diff(inst_phase)] / (2*pi*cfg.Ts);
    active = false(cfg.N,1);
    active(p.start_idx:p.stop_idx) = true;
    inst_freq(~active) = NaN;
    plot(t, inst_freq/1e6, 'w', 'LineWidth', 1.4); xlabel('t [\mus]'); ylabel('f_{inst} [MHz]');
    title(sprintf('Inst. freq (%s)', p.variant)); grid on;

    subplot(6, 3, (k-1)*3 + 3);
    spectrogram(s, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title('STFT spectrogram');
end

fprintf('Variant mix in Figure 1: quadratic=%d, sinusoidal=%d\n', ...
        variant_count.quadratic, variant_count.sinusoidal);

% ------------------------------------------------------------------
% Figure 2: LFM vs NLFM-quadratic vs NLFM-sinusoidal
% Force one of each by repeated sampling
% ------------------------------------------------------------------
figure('Name', 'LFM vs NLFM variants', 'Position', [120 60 1300 600]);

% Generate one LFM
[s_lfm, p_lfm] = generate_lfm(cfg);

% Force one quadratic and one sinusoidal NLFM
s_quad = []; p_quad = [];
s_sin  = []; p_sin  = [];
for trial = 1:50
    [s_, p_] = generate_nlfm(cfg);
    if isempty(s_quad) && strcmp(p_.variant, 'quadratic')
        s_quad = s_; p_quad = p_;
    end
    if isempty(s_sin) && strcmp(p_.variant, 'sinusoidal')
        s_sin = s_; p_sin = p_;
    end
    if ~isempty(s_quad) && ~isempty(s_sin), break; end
end

triples = {
    s_lfm,  p_lfm,  'LFM (linear)';
    s_quad, p_quad, 'NLFM (quadratic)';
    s_sin,  p_sin,  'NLFM (sinusoidal)'
};

for k = 1:3
    s = triples{k,1}; p = triples{k,2}; name = triples{k,3};
    t = (0:cfg.N-1) * cfg.Ts * 1e6;

    subplot(3, 2, (k-1)*2 + 1);
    inst_phase = unwrap(angle(s));
    inst_freq  = [0; diff(inst_phase)] / (2*pi*cfg.Ts);
    active = false(cfg.N,1); active(p.start_idx:p.stop_idx) = true;
    inst_freq(~active) = NaN;
    plot(t, inst_freq/1e6, 'w', 'LineWidth', 1.5);
    xlabel('t [\mus]'); ylabel('f_{inst} [MHz]');
    title([name, '  -  Instantaneous frequency']); grid on;

    subplot(3, 2, (k-1)*2 + 2);
    spectrogram(s, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title([name, '  -  STFT spectrogram']);
end

% ------------------------------------------------------------------
% Figure 3: NLFM-quadratic at multiple SNRs
% ------------------------------------------------------------------
figure('Name', 'NLFM at different SNRs', 'Position', [180 60 1200 700]);
snr_test = [-10, 0, 10, 20];

for i = 1:numel(snr_test)
    s_noisy = add_awgn(s_quad, snr_test(i), [p_quad.start_idx, p_quad.stop_idx]);

    subplot(numel(snr_test), 2, 2*i - 1);
    t = (0:cfg.N-1) * cfg.Ts * 1e6;
    plot(t, real(s_noisy));
    title(sprintf('NLFM-quadratic, real part, SNR = %d dB', snr_test(i)));
    xlabel('t [\mus]'); ylabel('amp'); grid on;

    subplot(numel(snr_test), 2, 2*i);
    spectrogram(s_noisy, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title(sprintf('NLFM-quadratic spectrogram, SNR = %d dB', snr_test(i)));
end

fprintf('Tests complete. Inspect three figures.\n');
fprintf('  - Figure 1: variant mix should be roughly 60/40 over many runs.\n');
fprintf('  - Figure 2: LFM=line, quadratic=parabola, sinusoidal=S-curve.\n');
fprintf('  - Figure 3: signal should disappear at SNR = -10 dB.\n');
