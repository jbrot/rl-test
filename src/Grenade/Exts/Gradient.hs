{-# LANGUAGE DataKinds, FlexibleContexts, FlexibleInstances, TypeFamilies, TypeOperators, UndecidableInstances #-}
module Grenade.Exts.Gradient where

import Data.Maybe (fromJust)
import Data.MonoTraversable (MonoFunctor(..), Element)
import Data.Singletons.TypeLits
import Data.VectorSpace
import Grenade
import Numeric.LinearAlgebra.Static
import qualified Numeric.LinearAlgebra as L
import qualified Numeric.LinearAlgebra.Devel as L

class MonoFunctor f => MonoApplicative f where
    opure :: Element f -> f
    oliftA2 :: (Element f -> Element f -> Element f) -> f -> f -> f

zipMats :: (KnownNat i, KnownNat o) => (Double -> Double -> Double) -> L i o -> L i o -> L i o
zipMats f a b = fromJust . create . L.reshape (snd $ size a) $ L.zipVectorWith f (L.flatten $ extract a) (L.flatten $ extract b)


type instance Element (FullyConnected' i o) = Double
instance (KnownNat i, KnownNat o) => MonoFunctor (FullyConnected' i o) where
    omap f (FullyConnected' b a)  = FullyConnected' (dvmap f b) (dmmap f a)
instance (KnownNat i, KnownNat o) => MonoApplicative (FullyConnected' i o) where
    opure c = FullyConnected' (konst c) (konst c)
    oliftA2 f (FullyConnected' a b) (FullyConnected' a2 b2) =  FullyConnected' (zipWithVector f a a2) (zipMats f b b2)
instance (KnownNat i, KnownNat o) => AdditiveGroup (FullyConnected' i o) where
    zeroV = opure 0
    (^+^) = oliftA2 (+)
    negateV = omap negate
instance (KnownNat i, KnownNat o) => VectorSpace (FullyConnected' i o) where
    type Scalar (FullyConnected' i o) = Double
    c *^ v = omap (c *) v

type instance Element (Gradients '[]) = Double
instance MonoFunctor (Gradients '[]) where
    omap f GNil = GNil
instance MonoApplicative (Gradients '[]) where
    opure _ = GNil
    oliftA2 _ GNil GNil = GNil

type instance Element (Gradients (a ': as)) = Element (Gradient a)
instance (MonoFunctor (Gradients as), MonoFunctor (Gradient a), UpdateLayer a, Element (Gradients as) ~ Element (Gradient a)) => MonoFunctor (Gradients (a ': as)) where
    omap f (a :/> b) = (omap f a) :/> (omap f b)
instance (MonoApplicative (Gradients as), MonoApplicative (Gradient a), UpdateLayer a, Element (Gradients as) ~ Element (Gradient a)) => MonoApplicative (Gradients (a ': as)) where
    opure c = (opure c) :/> (opure c)
    oliftA2 f (a :/> b) (a2 :/> b2) = (oliftA2 f a a2) :/> (oliftA2 f b b2)

instance (MonoApplicative (Gradients as), Num (Element (Gradients as))) => AdditiveGroup (Gradients as) where
    zeroV = opure 0
    (^+^) = oliftA2 (+)
    negateV = omap negate
instance (AdditiveGroup (Gradients as), MonoFunctor (Gradients as), Num (Element (Gradients as))) => VectorSpace (Gradients as) where
    type Scalar (Gradients as) = Element (Gradients as)
    c *^ v = omap (c *) v

type instance Element () = Double
instance MonoFunctor () where
    omap f () = ()
instance MonoApplicative () where
    opure _ = ()
    oliftA2 _ () () = ()
instance VectorSpace () where
    type Scalar () = Double
    c *^ v = omap (c *) v
