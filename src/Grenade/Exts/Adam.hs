{-# LANGUAGE FlexibleContexts, ScopedTypeVariables, TypeFamilies #-}
module Grenade.Exts.Adam where

import Data.MonoTraversable (MonoFunctor(..), Element)
import Data.VectorSpace
import Grenade
import Grenade.Exts.Gradient
import Grenade.Exts.Layer

data Adam layers = Adam { alpha :: Element (Gradients layers)
                        , beta1 :: Element (Gradients layers)
                        , beta2 :: Element (Gradients layers)
                        , epsilon :: Element (Gradients layers)
                        , mom :: Gradients layers
                        , vel :: Gradients layers
                        , time :: Int
                        }

defAdam :: forall layers. ( AdditiveGroup (Gradients layers)
                          , Fractional (Element (Gradients layers))
                          ) => Adam layers
defAdam = Adam (rtf 0.001) (rtf 0.9) (rtf 0.999) (rtf 1e-8) zeroV zeroV 0
    where rtf :: Double -> Element (Gradients layers)
          rtf = realToFrac

runAdam :: ( MonoApplicative (Gradients layers)
           , Floating (Element (Gradients layers))
           , All UpdateLayerRaw layers
           ) => Adam layers -> Gradients layers -> Network layers shapes -> (Adam layers, Network layers shapes)
runAdam a g n = (a{mom = m, vel = v, time = t}, applyRaw del n)
  where t = 1 + (time a)
        m = (beta1 a) *^ (mom a) ^+^ (1 - beta1 a) *^ g
        v = (beta2 a) *^ (vel a) ^+^ (1 - beta2 a) *^ (omap (^2) g)
        at = (alpha a) * sqrt (1 - (beta2 a)^t) / (1 - (beta1 a)^t)
        del = oliftA2 (\x y -> (-at) * x / (sqrt y + epsilon a)) m v
