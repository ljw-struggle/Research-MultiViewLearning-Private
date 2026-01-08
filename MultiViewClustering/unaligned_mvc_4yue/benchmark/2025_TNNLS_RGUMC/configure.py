def get_default_config(data_name):
    if data_name in ['Caltech101-20']:
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=1, #0.5
                start_dual_prediction=100,
                batch_size=256,
                epoch=50,# 20
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
                lambda3=0.01,
            ),
        )

    elif data_name in ['Scene_15']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[20, 1024, 1024, 1024, 128],
                arch2=[59, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5
                seed=8,
                start_dual_prediction=100,
                batch_size=256,
                epoch=20,#800,
                lr=1.0e-4,#1.0e-4,
                alpha=9,#[0.1, 0.5, 1, 5, 1e1],#9,
                lambda1=0.1,#[1e-3, 1e-1, 1, 1e1, 1e3],#0.1,
                lambda2=0.1,#[1e-3, 1e-1, 1, 1e1, 1e3],#0.1,
            ),
        )

    elif data_name in ['NoisyMNIST']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 64],
                arch2=[784, 1024, 1024, 1024, 64],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5
                seed=0,
                start_dual_prediction=100,
                epoch=20,#500,
                batch_size=256,
                lr=1.0e-4,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )

    elif data_name in ['LandUse_21']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[59, 1024, 1024, 1024, 64],
                arch2=[40, 1024, 1024, 1024, 64],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,
                seed=3,
                start_dual_prediction=100,
                epoch=20,#500,
                batch_size=256,
                lr=1.0e-4,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )


    elif data_name in ['digit_2view']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[76, 1024, 1024, 1024, 128],
                arch2=[216, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                epoch=20,  # 500,
                batch_size=256,
                lr=1.0e-4,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,

            ),
        )

    elif data_name in ['flower17_2views']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1360, 1024, 1024, 1024, 128],
                arch2=[1360, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,#4485,#256,#2048,
                epoch=20,#500,
                lr=1.0e-4,
                alpha=9,#The entropy parameter α
                lambda1=0.1,  # [1e-3, 1e-1, 1, 1e1, 1e3],#0.1,
                lambda2=0.1,  # [1e-3, 1e-1, 1, 1e1, 1e3],#0.1,
            ),
        )

    elif data_name in ['reuters_2views']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[21526, 1024, 1024, 1024, 128],
                arch2=[24892, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,  # 4485,#256,#2048,
                epoch=20,  # 500,
                lr=1.0e-4,
                alpha=9,  # The entropy parameter α
                lambda1=0.1,  # [1e-3, 1e-1, 1, 1e1, 1e3],#0.1,
                lambda2=0.1,  # [1e-3, 1e-1, 1, 1e1, 1e3],#0.1,
            ),
        )

    elif data_name in ['reuters_views_18758']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[21531, 1024, 1024, 1024, 128],
                arch2=[24892, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=2048,  # 4485,#256,#2048,
                epoch=20,  # 500,
                lr=1.0e-4,
                alpha=9,  # The entropy parameter α
                lambda1=0.1,  # [1e-3, 1e-1, 1, 1e1, 1e3],#0.1,
                lambda2=0.1,  # [1e-3, 1e-1, 1, 1e1, 1e3],#0.1,
            ),
        )



    elif data_name in ['AWA_2views']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[2688, 1024, 1024, 1024, 128],
                arch2=[2000, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=2048,  # 4485,#256,#2048,
                epoch=20,  # 500,
                lr=1.0e-4,
                alpha=9,  # The entropy parameter α
                lambda1=0.1,  # [1e-3, 1e-1, 1, 1e1, 1e3],#0.1,
                lambda2=0.1,  # [1e-3, 1e-1, 1, 1e1, 1e3],#0.1,
            ),
        )


    elif data_name in ['YouTuBe']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[2000, 1024, 1024, 1024, 128],
                arch2=[1000, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,  # 4485,#256,#2048,
                epoch=20,  # 500,
                lr=1.0e-4,
                alpha=9,  # The entropy parameter α
                lambda1=0.1,  # [1e-3, 1e-1, 1, 1e1, 1e3],#0.1,
                lambda2=0.1,  # [1e-3, 1e-1, 1, 1e1, 1e3],#0.1,
            ),
        )
    else:
        raise Exception('Undefined data_name')
