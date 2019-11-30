{-# LANGUAGE DataKinds, FlexibleContexts, FlexibleInstances, GADTs, ScopedTypeVariables, TypeFamilies, TypeOperators, UndecidableInstances #-}
module NN where

import Control.Monad.IO.Class
import Data.Maybe (fromJust)
import Data.Proxy
import Data.Singletons
import Data.Singletons.Prelude.Bool
import Data.Singletons.Prelude.Num
import Data.Singletons.TypeLits
import Data.Type.Equality ((:~:)(..))
import qualified Data.Vector.Storable as V
import Data.VectorSpace
import Grenade
import Numeric.LinearAlgebra.Static
import System.Random

type NL = '[ FullyConnected 4 128, Relu, FullyConnected 128 2, Softmax ]
type NNet = Network NL '[ 'D1 4, 'D1 128, 'D1 128, 'D1 2, 'D1 2 ]

randNN :: IO NNet
randNN = randomNetwork

sample :: (MonadIO m, KnownNat n, (1 <=? n) ~ 'True) => R n -> m Int
sample v = fmap (go v) . liftIO . randomRIO $ (0,1)
    where go :: forall n1. (KnownNat n1, (1 <=? n1) ~ 'True) => R n1 -> Double -> Int
          go vec v = if v < h
                        then 0
                        else case (SNat :: SNat n1) %- (SNat :: SNat 1) of
                               SNat -> case (SNat :: SNat 1) %<=? singByProxy t of
                                         STrue -> 1 + go t (v - h)
                                         SFalse -> 0
              where (h,t) = headTail vec

apply :: MonadIO m => NNet -> R 4 -> m (Int, Gradients NL)
apply nn v = fmap (\t -> (t, fst . runGradient nn tape . S1D . fromJust . create . flip V.unsafeUpd [(t,1)] $ V.replicate 2 0)) (sample o)
    where (tape, S1D o) = runNetwork nn (S1D v)

-- Orphan instances to make Gradients a vector space.

instance (KnownNat i, KnownNat o) => AdditiveGroup (FullyConnected' i o) where
    zeroV = FullyConnected' (konst 0) (konst 0)
    (FullyConnected' b a) ^+^ (FullyConnected' b2 a2) = FullyConnected' (b + b2) (a + a2)
    negateV (FullyConnected' b a) = FullyConnected' (-b) (-a)
instance (KnownNat i, KnownNat o) => VectorSpace (FullyConnected' i o) where
    type Scalar (FullyConnected' i o) = Double
    c *^ (FullyConnected' b a) = FullyConnected' (dvmap (*c) b) (dmmap (*c) a)

instance AdditiveGroup (Gradients '[]) where
    zeroV = GNil
    GNil ^+^ GNil = GNil
    negateV GNil = GNil
instance VectorSpace (Gradients '[]) where
    type Scalar (Gradients '[]) = Double
    c *^ GNil = GNil

instance (AdditiveGroup (Gradients as), AdditiveGroup (Gradient a), UpdateLayer a) => AdditiveGroup (Gradients (a ': as)) where
    zeroV = zeroV :/> zeroV
    (a :/> b) ^+^ (c :/> d) = (a ^+^ c) :/> (b ^+^ d)
    negateV (a :/> b) = (negateV a) :/> (negateV b)
instance (VectorSpace (Gradients as), VectorSpace (Gradient a), Scalar(Gradient a) ~ Scalar(Gradients as), UpdateLayer a) => VectorSpace (Gradients (a ': as)) where
    type Scalar (Gradients (a ': as)) = Scalar (Gradients as)
    c *^ (a :/> b) = (c *^ a) :/> (c *^ b)
instance VectorSpace () where
    type Scalar () = Double
    c *^ () = ()
