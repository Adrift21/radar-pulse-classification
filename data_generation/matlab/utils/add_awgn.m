function [noisy, noise_power] = add_awgn(signal, snr_db, active_idx)
% ADD_AWGN  Add complex AWGN at a specified SNR (active-region power).
%
% Inputs
% ------
%   signal     : (N x 1) complex signal (typically already padded)
%   snr_db     : desired SNR in dB
%   active_idx : (optional) [start_idx, stop_idx] indices defining the
%                active (non-zero pulse) region used for signal power.
%                If omitted, signal power is computed over the full N.
%
% Outputs
% -------
%   noisy       : (N x 1) signal + AWGN, same length as input
%   noise_power : variance of the added noise (linear, total complex)
%
% Convention
% ----------
% SNR is defined relative to the signal power on the *active region*
% only. This way, padding zeros do not artificially deflate the apparent
% signal power and inflate the noise needed for a target SNR.
%
% For complex AWGN: variance per sample is split equally between real
% and imaginary components, so each is N(0, noise_power/2).

    signal = signal(:);
    N = numel(signal);

    if nargin < 3 || isempty(active_idx)
        s_active = signal;
    else
        s_active = signal(active_idx(1):active_idx(2));
    end

    sig_power = mean(abs(s_active).^2);

    if sig_power <= 0
        error('add_awgn:zeroSignal', ...
              'Signal has zero power on active region; cannot set SNR.');
    end

    snr_linear  = 10^(snr_db / 10);
    noise_power = sig_power / snr_linear;        % total complex variance

    % Complex AWGN: real & imag each have variance noise_power/2
    noise = sqrt(noise_power/2) * (randn(N,1) + 1j * randn(N,1));

    noisy = signal + noise;
end
