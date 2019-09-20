%Autor: Denilson Gomes Vaz da Silva <denilsongomes@alu.ufc.br>
%Graduando em Engenharia da Computação
%Reconhecimento de padroes

%Algoritmo para classificar base de dados usando o K-NN
clc
close all
clear

%Carrega base de dados
load('Classe1.mat')
load('Classe2.mat')
Y1 = Classe1';
Y2 = Classe2';
dataset = ones(100,501);
dataset(1:50,1:end-1) = Y1;
dataset(51:end,1:end-1) = Y2;

%atribui o rotulo de cada classe
dataset(51:end,end) = 2;

%Plotagem dos dados
plot(Y1, 'r')
hold on
plot(Y2, 'b')
%TF do sinal
hold off

for i=1:100   
    Y(i,1:500)=(abs(fftshift(fft(dataset(i,1:500)))));
end
freq_vec = linspace(-pi,pi,length(dataset(1,1:500)));
figure,plot(freq_vec,abs(Y(2,1:500)),'b')
figure,plot(freq_vec,abs(Y(52,1:500)),'b')

%Amplitude Maxima do sinal no tempo
%Amplitude Media do sinal no tempo
%Desvio Padrão do sinal no tempo
for i=1:100
    AmpMax(i) = max(dataset(i,1:500));
    AmpMedia(i) = mean(dataset(i,1:500));
    DesvioAmp(i) = std(dataset(i,1:500));
end
%Atributos da classe 1
disp('Atributos da classe 1')
AmpMax1 = mean(AmpMax(1:50))
AmpMedia1 = mean(AmpMedia(1:50))
DesvioAmp1 = mean(DesvioAmp(1:50))

%Atributos da classe 2
disp('Atributos da classe 2')
AmpMax2 = mean(AmpMax(51:end))
AmpMedia1 = mean(AmpMedia(51:end))
DesvioAmp1 = mean(DesvioAmp(51:end))


