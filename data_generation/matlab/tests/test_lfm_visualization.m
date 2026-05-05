% TEST_LFM_VISUALIZATION
% Quick sanity check: generate a few LFM pulses, plot waveform, instantaneous
% frequency, and STFT spectrogram. Also verify SNR addition visually.
%
% Run from data_generation/matlab/ directory:
%   >> cd data_generation/matlab
%   >> addpath(genpath(pwd))
%   >> test_lfm_visualization

clear; clc; close all;

% Make sure subfolders are on the path
addpath('config', 'signals', 'utils');

cfg = generation_config();

% ------------------------------------------------------------------
% Generate 4 random LFM pulses (clean) and plot
% ------------------------------------------------------------------
figure('Name', 'LFM clean pulses', 'Position', [100 100 1200 700]);

for k = 1:4
    [s, p] = generate_lfm(cfg);

    % Time axis in microseconds
    t = (0:cfg.N-1) * cfg.Ts * 1e6;

    subplot(4, 3, (k-1)*3 + 1);
    plot(t, real(s), 'b'); hold on; plot(t, imag(s), 'r');
    xlabel('t [\mus]'); ylabel('amp');
    title(sprintf('LFM #%d  T=%.2f \\mus  B=%.1f MHz  %s', ...
          k, p.pulse_width_s*1e6, p.bandwidth_hz/1e6, p.direction));
    legend('I','Q'); grid on;

    % Instantaneous frequency from phase derivative
    subplot(4, 3, (k-1)*3 + 2);
    inst_phase = unwrap(angle(s));
    inst_freq  = [0; diff(inst_phase)] / (2*pi*cfg.Ts);
    % Mask to active region for clarity
    active = false(cfg.N,1);
    active(p.start_idx:p.stop_idx) = true;
    inst_freq(~active) = NaN;
    plot(t, inst_freq/1e6, 'w', 'LineWidth', 1.4); xlabel('t [\mus]'); ylabel('f_{inst} [MHz]');
    title('Instantaneous frequency'); grid on;

    % STFT spectrogram
    subplot(4, 3, (k-1)*3 + 3);
    spectrogram(s, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title('STFT spectrogram');
end

% ------------------------------------------------------------------
% Verify AWGN: same signal at multiple SNRs
% ------------------------------------------------------------------
[s_clean, p] = generate_lfm(cfg);
snr_test = [-10, 0, 10, 20];

figure('Name', 'AWGN at different SNRs', 'Position', [200 100 1200 700]);
for i = 1:numel(snr_test)
    s_noisy = add_awgn(s_clean, snr_test(i), [p.start_idx, p.stop_idx]);

    subplot(numel(snr_test), 2, 2*i - 1);
    t = (0:cfg.N-1) * cfg.Ts * 1e6;
    plot(t, real(s_noisy));
    title(sprintf('Real part, SNR = %d dB', snr_test(i)));
    xlabel('t [\mus]'); ylabel('amp'); grid on;

    subplot(numel(snr_test), 2, 2*i);
    spectrogram(s_noisy, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title(sprintf('Spectrogram, SNR = %d dB', snr_test(i)));
end

fprintf('Tests complete. Inspect both figures.\n');
fprintf('  - LFM figure: chirp ramp should be visible in spectrogram.\n');
fprintf('  - AWGN figure: signal should disappear into noise as SNR drops.\n');
