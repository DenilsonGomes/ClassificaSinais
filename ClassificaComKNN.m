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
    MediaFurrier(i) = mean(abs(YTF(i,:)));
    MedianaFurrier(i) = median(abs(YTF(i,:)));
    skew(i) = (sum((YTF(i,:)-mean(YTF(i,:))).^3)./length(YTF(i,:)))./ (var(YTF(i,:),1).^1.5);
    kurt(i) = (sum((YTF(i,:)-mean(YTF(i,:))).^4)./length(YTF(i,:))) ./ (var(YTF(i,:),1).^2);
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
dados = ones(100,9);
dados(i,1) = max(dataset(i,1:500));
dados(i,2) = mean(dataset(i,1:500));
dados(i,3) = std(dataset(i,1:500));
dados(i,4) = kurtosis(YTF(i,:));
dados(i,5) = skewness(YTF(i,:));
dados(i,2) = max(YTF(i,:));
dados(i,3) = mean(YTF(i,:));
dados(i,4) = median(YTF(i,:));
dados(51:end,9) = 2;

%k-Fold
k=5
for i=1:k
    %pega 5 amostras de cada classe
    Teste(1:5,:)=dados(1+(i-1)*5:i*5,:);
    Teste(6:10,:)=dados(51+(i-1)*5:50+i*5,:);
    dados(1+(i-1)*5:i*5,:)=[];
    dados(51+(i-1)*5-5:50+i*5-5,:)=[];
   
    
  
    
    for g=1:10
        aux5=zeros(5,2);
        Treino=Atributos;
        for k=1:90
    aux(k)=norm(Teste(g,1:num_atributos)-Treino(k,1:num_atributos)); %media euclidiana
        end 
        
        
        
        for K=1:5 %5NN
        [M N]=min(aux);%pega índice da menor distância
        aux5(K,:)=Treino(N,num_atributos+1:end); %pega o rótulo da amostra mais próxima
        Treino(N,:)=[];
        aux(K)=[];
        end
             aux2(1)=mean(aux5(:,1));
             aux2(2)=mean(aux5(:,2));
        
        if aux2(1)>aux2(2)
            aux2=[1 0];
        else
            aux2=[0 1];
        end
        
        if aux2==Teste(g,num_atributos+1:end);
            aux3=aux3+1;
        else 
            aux4=aux4+1;
        end
    end
    
end