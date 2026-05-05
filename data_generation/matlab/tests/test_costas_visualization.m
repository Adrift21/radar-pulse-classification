% TEST_COSTAS_VISUALIZATION
% Sanity check for Costas frequency-hopping generator. Produces:
%   - Figure 1: 6 Costas pulses (mix of N=5/6/7/8).
%               Shows real part, instantaneous frequency, STFT spectrogram
%               with hop frequencies overlaid as dashed lines.
%   - Figure 2: Force one of each N (5, 6, 7, 8) for direct comparison.
%               Includes the Costas time-frequency "thumbtack pattern".
%   - Figure 3: 6-class comparison: LFM, NLFM, Barker, Frank, Polyphase,
%               Costas — Costas should look nothing like the others.
%   - Figure 4: Costas at multiple SNRs (the hops should disappear at
%               low SNR).
%
% Run from data_generation/matlab/ directory:
%   >> cd data_generation/matlab
%   >> addpath(genpath(pwd))
%   >> test_costas_visualization

clear; clc; close all;

addpath('config', 'signals', 'utils');

cfg = generation_config();

% ------------------------------------------------------------------
% Figure 1: 6 Costas pulses (clean)
% ------------------------------------------------------------------
figure('Name', 'Costas clean pulses', 'Position', [60 60 1300 800]);

N_count = struct('N5', 0, 'N6', 0, 'N7', 0, 'N8', 0);

for k = 1:6
    [s, p] = generate_costas(cfg);
    N_count.(sprintf('N%d', p.N)) = N_count.(sprintf('N%d', p.N)) + 1;

    t = (0:cfg.N-1) * cfg.Ts * 1e6;

    subplot(6, 3, (k-1)*3 + 1);
    plot(t, real(s), 'b'); hold on; plot(t, imag(s), 'r');
    xlabel('t [\mus]'); ylabel('amp');
    title(sprintf('Costas #%d  N=%d  T=%.2f \\mus  T_c=%.0f ns  \\Delta f=%.1f MHz', ...
          k, p.N, p.pulse_width_s*1e6, ...
          p.chip_duration_s*1e9, p.delta_f_hz/1e6));
    legend('I','Q', 'Location', 'eastoutside'); grid on;

    % Instantaneous frequency from phase derivative
    subplot(6, 3, (k-1)*3 + 2);
    inst_phase = unwrap(angle(s));
    inst_freq  = [0; diff(inst_phase)] / (2*pi*cfg.Ts);
    active = false(cfg.N,1);
    active(p.start_idx:p.stop_idx) = true;
    inst_freq(~active) = NaN;
    plot(t, inst_freq/1e6, 'w', 'LineWidth', 1.4); hold on;
    % Overlay the expected hop frequencies as horizontal dashed lines
    for f_hop = p.frequencies_hz
        yline(f_hop/1e6, 'Color', [0.4 0.9 0.4], ...
              'LineStyle', '--', 'Alpha', 0.7);
    end
    xlabel('t [\mus]'); ylabel('f_{inst} [MHz]');
    title(sprintf('Inst. freq (N=%d hops)', p.N));
    grid on;

    subplot(6, 3, (k-1)*3 + 3);
    spectrogram(s, hamming(64), 56, 256, cfg.fs, 'yaxis', 'centered');
    title('STFT spectrogram');
end

fprintf('N mix in Figure 1: N=5 -> %d, N=6 -> %d, N=7 -> %d, N=8 -> %d\n', ...
        N_count.N5, N_count.N6, N_count.N7, N_count.N8);

% ------------------------------------------------------------------
% Figure 2: Force one example of each N for direct comparison
% ------------------------------------------------------------------
figure('Name', 'Costas N=5/6/7/8 comparison', 'Position', [120 30 1300 900]);

samples = struct();
target_N = [5, 6, 7, 8];

for trial = 1:200
    [s_, p_] = generate_costas(cfg);
    key = sprintf('N%d', p_.N);
    if ismember(p_.N, target_N) && ~isfield(samples, key)
        samples.(key) = struct('s', s_, 'p', p_);
    end
    if numel(fieldnames(samples)) == 4, break; end
end

for idx = 1:4
    Nval = target_N(idx);
    s = samples.(sprintf('N%d', Nval)).s;
    p = samples.(sprintf('N%d', Nval)).p;
    t = (0:cfg.N-1) * cfg.Ts * 1e6;

    % Costas sequence as discrete points
    subplot(4, 3, (idx-1)*3 + 1);
    stem(1:p.N, double(p.sequence), 'filled', 'MarkerSize', 6, ...
         'Color', [0.2 0.6 1]);
    xlabel('chip k'); ylabel('\pi(k)');
    title(sprintf('N=%d  sequence \\pi = [%s]', p.N, ...
          sprintf('%d ', p.sequence)));
    ylim([0, p.N+1]); xlim([0, p.N+1]);
    set(gca, 'YTick', 1:p.N, 'XTick', 1:p.N);
    grid on;

    % Hop frequencies (these are pi(k) - (N+1)/2 * delta_f)
    subplot(4, 3, (idx-1)*3 + 2);
    inst_phase = unwrap(angle(s));
    inst_freq  = [0; diff(inst_phase)] / (2*pi*cfg.Ts);
    active = false(cfg.N,1); active(p.start_idx:p.stop_idx) = true;
    inst_freq(~active) = NaN;
    plot(t, inst_freq/1e6, 'w', 'LineWidth', 1.6); hold on;
    for f_hop = p.frequencies_hz
        yline(f_hop/1e6, 'Color', [0.4 0.9 0.4], ...
              'LineStyle', '--', 'Alpha', 0.7);
    end
    xlabel('t [\mus]'); ylabel('f_{inst} [MHz]');
    title(sprintf('Inst. freq, \\Delta f=%.1f MHz', p.delta_f_hz/1e6));
    grid on;

    % Spectrogram (the Costas thumbtack pattern!)
    subplot(4, 3, (idx-1)*3 + 3);
    spectrogram(s, hamming(64), 56, 256, cfg.fs, 'yaxis', 'centered');
    title(sprintf('N=%d  -  STFT spectrogram', Nval));
end

% ------------------------------------------------------------------
% Figure 3: 6-class comparison
% ------------------------------------------------------------------
figure('Name', '6-class comparison', 'Position', [180 20 1300 950]);

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

sextet = {
    s_lfm,  p_lfm,  'LFM (linear)';
    s_nlfm, p_nlfm, 'NLFM (quadratic)';
    s_brk,  p_brk,  sprintf('Barker (%s)', p_brk.code_name);
    s_frk,  p_frk,  sprintf('Frank (N=%d)', p_frk.N);
    s_pp,   p_pp,   sprintf('Polyphase (%s, N=%d)', p_pp.subcode, p_pp.N);
    s_cos,  p_cos,  sprintf('Costas (N=%d)', p_cos.N)
};

for k = 1:6
    s = sextet{k,1}; p = sextet{k,2}; name = sextet{k,3};

    subplot(6, 2, (k-1)*2 + 1);
    inst_phase = unwrap(angle(s));
    inst_freq  = [0; diff(inst_phase)] / (2*pi*cfg.Ts);
    active = false(cfg.N,1); active(p.start_idx:p.stop_idx) = true;
    inst_freq(~active) = NaN;
    plot((0:cfg.N-1)*cfg.Ts*1e6, inst_freq/1e6, 'w', 'LineWidth', 1.2);
    xlabel('t [\mus]'); ylabel('f_{inst} [MHz]');
    title([name, '  -  Instantaneous frequency']);
    grid on;

    subplot(6, 2, (k-1)*2 + 2);
    spectrogram(s, hamming(64), 56, 256, cfg.fs, 'yaxis', 'centered');
    title([name, '  -  STFT spectrogram']);
end

% ------------------------------------------------------------------
% Figure 4: Costas at multiple SNRs
% ------------------------------------------------------------------
figure('Name', 'Costas at different SNRs', 'Position', [240 60 1200 700]);
snr_test = [-10, 0, 10, 20];

for i = 1:numel(snr_test)
    s_noisy = add_awgn(s_cos, snr_test(i), [p_cos.start_idx, p_cos.stop_idx]);

    subplot(numel(snr_test), 2, 2*i - 1);
    t = (0:cfg.N-1) * cfg.Ts * 1e6;
    plot(t, real(s_noisy));
    title(sprintf('Costas (N=%d), real part, SNR = %d dB', p_cos.N, snr_test(i)));
    xlabel('t [\mus]'); ylabel('amp'); grid on;

    subplot(numel(snr_test), 2, 2*i);
    spectrogram(s_noisy, hamming(64), 56, 256, cfg.fs, 'yaxis', 'centered');
    title(sprintf('Costas spectrogram, SNR = %d dB', snr_test(i)));
end

fprintf('Tests complete. Inspect four figures.\n');
fprintf('  - Figure 1: 6 Costas pulses; inst. freq should be N step plateaus.\n');
fprintf('  - Figure 2: stem shows the permutation; spectrogram = thumbtack pattern.\n');
fprintf('  - Figure 3: Costas spectrogram = N short blocks (unique signature).\n');
fprintf('  - Figure 4: hop blocks should fade into noise at SNR = -10 dB.\n');