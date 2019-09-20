%Autor: Denilson Gomes Vaz da Silva <denilsongomes@alu.ufc.br>
%Graduando em Engenharia da Computação
%Reconhecimento de padroes

%Algoritmo para classificar base de dados usando o K-NN
%Os valores para kfold e kvizinhos devem ser alterado manualemente
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

%atribui o rotulo de cada classe2
dataset(51:end,end) = 2;

%Estudo dos dados

%Plotagem dos dados
plot(Y1, 'r')
hold on
plot(Y2, 'b')
hold off

%Amplitude Maxima do sinal no tempo
%Amplitude Media do sinal no tempo
%Desvio Padrão do sinal no tempo
for i=1:100
    AmpMax(i) = max(dataset(i,1:500));
    AmpMedia(i) = mean(dataset(i,1:500));
    DesvioAmp(i) = std(dataset(i,1:500));
end
% %Atributos da classe 1
% disp('Atributos da classe 1')
% AmpMax1 = mean(AmpMax(1:50))
% AmpMedia1 = mean(AmpMedia(1:50))
% DesvioAmp1 = mean(DesvioAmp(1:50))
% 
% %Atributos da classe 2
% disp('Atributos da classe 2')
% AmpMax2 = mean(AmpMax(51:end))
% AmpMedia1 = mean(AmpMedia(51:end))
% DesvioAmp1 = mean(DesvioAmp(51:end))

%TF do sinal
for i=1:100   
    YTF(i,1:500)=(abs(fftshift(fft(dataset(i,1:500)))));
end
freq_vec = linspace(-pi,pi,length(dataset(1,1:500)));
% figure,plot(freq_vec,abs(YTF(1,1:500)),'b')
% figure,plot(freq_vec,abs(YTF(62,1:500)),'b')

%Kurtoise
%Skewness
%maximo Modulo da TF
%media
%mediana
for i=1:100   
    ModuloFurrier(i) = max(abs(YTF(i,:)));
    skew(i) = (sum((YTF(i,:)-mean(YTF(i,:))).^3)./length(YTF(i,:)))./ (var(YTF(i,:),1).^1.5);
    kurt(i) = (sum((YTF(i,:)-mean(YTF(i,:))).^4)./length(YTF(i,:))) ./ (var(YTF(i,:),1).^2);
    if (i <=50)
        MediaFurrier1(i) = mean(abs(YTF(i,1:50)));
        MedianaFurrier1(i) = median(abs(YTF(i,1:50)));
    end
    if (i>50)
        MediaFurrier2(i) = mean(abs(YTF(i,51:100)));
        MedianaFurrier2(i) = median(abs(YTF(i,51:100)));
    end
end
% MaxModuloFurrier1 = mean(ModuloFurrier(1:50))
% MaxModuloFurrier2 = mean(ModuloFurrier(51:end))
% 
% MediaFurrier1 = mean(MediaFurrier(1:50))
% MediaFurrier2 = mean(MediaFurrier(51:end))
% 
% MedianaFurrier1 = mean(MedianaFurrier(1:50))
% MedianaFurrier2 = mean(MedianaFurrier(51:end))

%Estes atributos serão usados na classificacao
%NO TEMPO {Amplitude Maxima, Amplitude Media, Desvio Padrão}
%NA FREQUENCIA {%Kurtoise, Skewness, maximo, media, mediana
%Logo, vamos construir nossa base de dados
dados(:,1) = AmpMax;
dados(:,2) = AmpMedia;
dados(:,3) = DesvioAmp;
dados(:,4) = kurt;
dados(:,5) = skew;
dados(:,6) = ModuloFurrier;
dados(1:50,7) = MediaFurrier1;
dados(51:100,7) = MediaFurrier2(51:end);
dados(1:50,8) = MedianaFurrier1;
dados(51:100,8) = MedianaFurrier2(51:end);
classes(1:50) = dataset(1:50,end);
classes(51:100) = dataset(51:end,end);

