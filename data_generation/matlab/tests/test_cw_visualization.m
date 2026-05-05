% TEST_CW_VISUALIZATION
% Sanity check for CW generator. Produces:
%   - Figure 1: 6 CW pulses with different fc values; shows real part,
%               instantaneous frequency (should be flat), STFT spectrogram
%               (should be a horizontal line at fc).
%   - Figure 2: 7-class comparison (LFM, NLFM, Barker, Frank, Polyphase,
%               Costas, CW). CW should look the simplest of all.
%   - Figure 3: CW at multiple SNRs; the single tone should fade into
%               noise from low SNR upward.
%
% Run from data_generation/matlab/ directory:
%   >> cd data_generation/matlab
%   >> addpath(genpath(pwd))
%   >> test_cw_visualization

clear; clc; close all;

addpath('config', 'signals', 'utils');

cfg = generation_config();

% ------------------------------------------------------------------
% Figure 1: 6 CW pulses with different fc values
% ------------------------------------------------------------------
figure('Name', 'CW clean pulses', 'Position', [60 60 1300 800]);

for k = 1:6
    [s, p] = generate_cw(cfg);

    t = (0:cfg.N-1) * cfg.Ts * 1e6;     % us

    subplot(6, 3, (k-1)*3 + 1);
    plot(t, real(s), 'b'); hold on; plot(t, imag(s), 'r');
    xlabel('t [\mus]'); ylabel('amp');
    title(sprintf('CW #%d  T=%.2f \\mus  f_c=%.1f MHz  \\phi_0=%.2f rad', ...
          k, p.pulse_width_s*1e6, p.f_carrier_hz/1e6, p.initial_phase));
    legend('I','Q', 'Location', 'eastoutside'); grid on;

    % Instantaneous frequency — should be FLAT at f_c
    subplot(6, 3, (k-1)*3 + 2);
    inst_phase = unwrap(angle(s));
    inst_freq  = [0; diff(inst_phase)] / (2*pi*cfg.Ts);
    active = false(cfg.N,1);
    active(p.start_idx:p.stop_idx) = true;
    inst_freq(~active) = NaN;
    plot(t, inst_freq/1e6, 'w', 'LineWidth', 1.4); hold on;
    yline(p.f_carrier_hz/1e6, 'Color', [0.4 0.9 0.4], ...
          'LineStyle', '--', 'Alpha', 0.7);
    xlabel('t [\mus]'); ylabel('f_{inst} [MHz]');
    title('Inst. freq (flat = expected)'); grid on;

    subplot(6, 3, (k-1)*3 + 3);
    spectrogram(s, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title('STFT spectrogram');
end

% ------------------------------------------------------------------
% Figure 2: 7-class comparison
% ------------------------------------------------------------------
figure('Name', '7-class comparison', 'Position', [120 20 1300 950]);

[s_lfm, p_lfm] = generate_lfm(cfg);

s_nlfm = []; p_nlfm = [];
for trial = 1:50
    [s_, p_] = generate_nlfm(cfg);
    if strcmp(p_.variant, 'quadratic'), s_nlfm=s_; p_nlfm=p_; break; end
end

[s_brk, p_brk] = generate_barker(cfg);
[s_frk, p_frk] = generate_frank(cfg);
[s_pp,  p_pp]  = generate_polyphase(cfg);
[s_cos, p_cos] = generate_costas(cfg);
[s_cw,  p_cw]  = generate_cw(cfg);

septet = {
    s_lfm,  p_lfm,  'LFM (linear)';
    s_nlfm, p_nlfm, 'NLFM (quadratic)';
    s_brk,  p_brk,  sprintf('Barker (%s)', p_brk.code_name);
    s_frk,  p_frk,  sprintf('Frank (N=%d)', p_frk.N);
    s_pp,   p_pp,   sprintf('Polyphase (%s, N=%d)', p_pp.subcode, p_pp.N);
    s_cos,  p_cos,  sprintf('Costas (N=%d)', p_cos.N);
    s_cw,   p_cw,   sprintf('CW (f_c=%.1f MHz)', p_cw.f_carrier_hz/1e6)
};

for k = 1:7
    s = septet{k,1}; p = septet{k,2}; name = septet{k,3};

    subplot(7, 2, (k-1)*2 + 1);
    inst_phase = unwrap(angle(s));
    inst_freq  = [0; diff(inst_phase)] / (2*pi*cfg.Ts);
    active = false(cfg.N,1); active(p.start_idx:p.stop_idx) = true;
    inst_freq(~active) = NaN;
    plot((0:cfg.N-1)*cfg.Ts*1e6, inst_freq/1e6, 'w', 'LineWidth', 1.0);
    xlabel('t [\mus]'); ylabel('f_{inst} [MHz]');
    title([name, '  -  Instantaneous frequency']);
    grid on;

    subplot(7, 2, (k-1)*2 + 2);
    spectrogram(s, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title([name, '  -  STFT spectrogram']);
end

% ------------------------------------------------------------------
% Figure 3: CW at multiple SNRs
% ------------------------------------------------------------------
figure('Name', 'CW at different SNRs', 'Position', [180 60 1200 700]);
snr_test = [-10, 0, 10, 20];

for i = 1:numel(snr_test)
    s_noisy = add_awgn(s_cw, snr_test(i), [p_cw.start_idx, p_cw.stop_idx]);

    subplot(numel(snr_test), 2, 2*i - 1);
    t = (0:cfg.N-1) * cfg.Ts * 1e6;
    plot(t, real(s_noisy));
    title(sprintf('CW, real part, SNR = %d dB', snr_test(i)));
    xlabel('t [\mus]'); ylabel('amp'); grid on;

    subplot(numel(snr_test), 2, 2*i);
    spectrogram(s_noisy, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title(sprintf('CW spectrogram (f_c=%.1f MHz), SNR = %d dB', ...
          p_cw.f_carrier_hz/1e6, snr_test(i)));
end

fprintf('Tests complete. Inspect three figures.\n');
fprintf('  - Figure 1: CW inst. freq must be flat at fc; spectrogram = single line.\n');
fprintf('  - Figure 2: 7-class comparison; CW is the simplest signature.\n');
fprintf('  - Figure 3: tone fades into noise at SNR = -10 dB.\n');
