{-# LANGUAGE ConstraintKinds, DataKinds, ScopedTypeVariables, TypeFamilies, TypeOperators #-}
module Main where

import Control.Monad
import Control.Monad.IO.Class
import Control.Monad.Trans.Control
import Control.Monad.Trans.State
import Data.MonoTraversable (MonoFunctor(..))
import Data.Singletons.TypeLits
import Data.VectorSpace
import GHC.Types (Constraint)
import Grenade
import Graphics.Gloss.Interface.IO.Simulate
import Graphics.Gloss.Data.ViewPort
import Numeric.LinearAlgebra.Static
import System.Random
import Text.Printf

import NN
import RL.Gym

data IState g = IState { gym :: g
                       , obs :: Observation g
                       , nn :: NNet
                       , rollout :: [(Float,Gradients NL)]
                       , adam :: Adam
                       , episode :: Int
                       , avg :: Float
                       }

data Adam = Adam { alpha :: Double
                 , beta1 :: Double
                 , beta2 :: Double
                 , epsilon :: Double
                 , mom :: Gradients NL
                 , vel :: Gradients NL
                 , time :: Int
                 }

defAdam :: Adam
defAdam = Adam 0.001 0.9 0.999 1e-8 zeroV zeroV 0

class UpdateLayer x => UpdateLayerRaw x where
    runUpdateRaw :: Gradient x -> x -> x

instance (KnownNat i, KnownNat o) => UpdateLayerRaw (FullyConnected i o) where
    runUpdateRaw d (FullyConnected a b) = FullyConnected (d ^+^ a) b
instance UpdateLayerRaw (Relu) where
    runUpdateRaw _ _ = Relu
instance UpdateLayerRaw (Softmax) where
    runUpdateRaw _ _ = Softmax

type family AllULR (as :: [*]) :: Constraint where
    AllULR '[] = ()
    AllULR (a ': as) = (UpdateLayerRaw a, AllULR as)

applyRaw :: AllULR layers => Gradients layers -> Network layers shapes -> Network layers shapes
applyRaw GNil NNil = NNil
applyRaw (g :/> gs) (n :~> ns) = (runUpdateRaw g n :~> ns)

runAdam :: Adam -> Gradients NL -> NNet -> (Adam, NNet)
runAdam a g n = (a{mom = m, vel = v, time = t}, applyRaw del n)
  where t = 1 + (time a)
        m = (beta1 a) *^ (mom a) ^+^ (1 - beta1 a) *^ g
        v = (beta2 a) *^ (vel a) ^+^ (1 - beta2 a) *^ (omap (^2) g)
        at = (alpha a) * sqrt (1 - (beta2 a)^t) / (1 - (beta1 a)^t)
        del = oliftA2 (\x y -> (-at) * x / (sqrt y + epsilon a)) m v

istate :: (Monad m) => (g -> (a,g)) -> StateT (IState g) m a
istate f = do
    (out, ngym)  <- fmap (f . gym) get
    modify $ \st -> st{gym = ngym}
    pure out

defSt :: Gym g => IO (IState g)
defSt = fmap reset start >>= \(o, g) -> IState g o <$> randNN <*> pure [] <*> pure defAdam <*> pure 0 <*> pure 0

rnd :: Gym g => IState g -> IO Picture
rnd = pure . translate (-300) (-200) . render 600 400 . gym

rtf = realToFrac

updateNet :: IState Cartpole -> IState Cartpole
updateNet st = st{rollout = [], adam = ad', nn = nn'}
  where gamma = 0.9
        (_,gtrl) = foldl (\(v,ls) (r,g) -> let nv = gamma * v + r in (nv, (nv,g):ls)) (0, []) (rollout st)
        average :: Fractional n => [n] -> n
        average = (/) <$> sum <*> (realToFrac . length)
        avg :: Float
        avg = average (fmap fst gtrl) 
        stdev :: Float
        stdev = sqrt . average . fmap (\(x,_) -> (x - avg)^2) $ gtrl
        upd = foldr (\(x,g) ag -> ag ^+^ ((rtf $ (x - avg) / (stdev + 1e-9)) *^ g)) zeroV gtrl
        (ad', nn') = runAdam (adam st) upd (nn st)

resetEp :: MonadIO m => StateT (IState Cartpole) m (Observation Cartpole)
resetEp = do
    s <- get
    let reward = sum . fmap fst . rollout $ s
        ep = 1 + episode s
        navg = 0.05 * reward + 0.95 * (avg s)
    put s{avg = navg, episode = ep}
    when (ep `mod` 10 == 0) . liftIO $ printf "Episode: %d Last reward: %.02f Average: %.02f\n" ep reward navg
    modify updateNet
    istate reset

stp :: ViewPort -> Float -> IState Cartpole -> IO (IState Cartpole)
stp _ _ = execStateT $ do
    CObs x1 x2 x3 x4 <- obs <$> get
    (act, grad) <- flip apply (vec4 (rtf x1) (rtf x2) (rtf x3) (rtf x4)) . nn =<< get
    (o, r, d) <- istate (step $ case act of
                                  0 -> CLeft
                                  1 -> CRight
                                  _ -> undefined)
    modify (\s -> s{rollout = (r,grad):(rollout s)})
    o <- if d then resetEp else pure o
    modify (\s -> s{obs = o})

main :: IO ()
main = do
    st <- defSt
    simulateIO (InWindow "Cartpole" (600,400) (10,10)) (makeColor 1 1 1 1) 500 st rnd stp
