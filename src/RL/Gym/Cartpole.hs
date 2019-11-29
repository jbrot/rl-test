{-# LANGUAGE FlexibleInstances, GeneralizedNewtypeDeriving, MultiParamTypeClasses, TypeFamilies, TypeSynonymInstances, UndecidableInstances #-}
module RL.Gym.Cartpole (Cartpole, Action (..), Observation (..)) where

import Control.Monad.IO.Class
import Control.Monad.Trans.Class
import Control.Monad.Trans.Control
import Control.Monad.Trans.State.Strict
import Data.Maybe
import Graphics.Gloss
import System.Random
import System.Random.Mersenne.Pure64

import RL.Gym.Types

data Cartpole = Cartpole { x :: Float, xdot :: Float, theta :: Float, thetadot :: Float, done :: Bool, gen :: Maybe PureMT }

gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = masspole + masscart
lgth = 0.5
polemass_length = masspole * lgth
force_mag = 10
tau = 0.02

theta_threshold_radians = 12 * 2 * pi / 360
x_threshold = 2.4

-- Given an action and a state, compute the new state
advance :: Action Cartpole -> Cartpole -> Cartpole
advance a (Cartpole xx xxd tt ttd d g) = Cartpole nxx nxxd ntt nttd d g
    where force = case a of
            CLeft -> -force_mag
            CRight -> force_mag
          -- Compute accelerations
          temp = (force + polemass_length * ttd * ttd * (sin tt)) / total_mass
          thetaacc = (gravity * (sin tt) - (cos tt) * temp) / (lgth * (4.0 / 3.0 - masspole * (cos tt) * (cos tt) / total_mass))
          xacc = temp - polemass_length * thetaacc * (cos tt) / total_mass
          -- Euler Integrate
          nxx = xx + tau * xxd
          nxxd = xxd + tau * xacc
          ntt = tt + tau * ttd
          nttd = ttd + tau * thetaacc

convert :: Cartpole -> Observation Cartpole
convert c = CObs (x c) (xdot c) (theta c) (thetadot c)

instance Gym Cartpole where
    data Action Cartpole = CLeft | CRight
    data Observation Cartpole = CObs { position :: Float -- [-4.8,4.8]
                                     , velocity :: Float -- (-Inf, Inf)
                                     , angle :: Float -- [-24, 24] degrees
                                     , velocityAtTip :: Float -- (-Inf, Inf)
                                     }
    step a st0 = ((convert st2, reward, dn), st2)
        where st1 = advance a st0
              dn = (abs . x $ st1) < x_threshold || (abs . theta $ st1) < theta_threshold_radians
              st2 = st1{done = dn}
              reward = if not dn || not (done st1) then 1 else 0

    start = fmap snd . seed Nothing $ Cartpole 0 0 0 0 False Nothing
    seed s st = do
        g <- case s of
               Just v -> pure (pureMT v)
               Nothing -> liftIO newPureMT
        pure ((), st{gen = Just g})
    reset st0 = (CObs p1 p2 p3 p4, Cartpole p1 p2 p3 p4 False (Just g4))
        where g = fromJust . gen $ st0
              (p1,g1) = randomR (-0.05,0.05) g
              (p2,g2) = randomR (-0.05,0.05) g1
              (p3,g3) = randomR (-0.05,0.05) g2
              (p4,g4) = randomR (-0.05,0.05) g3
    render w h st = pic
        where screen_width = 600
              screen_height = 400
              world_width = x_threshold * 2
              scl = screen_width / world_width
              carty = 100
              polewidth = 10
              polelen = scl * 2 * lgth
              cartwidth = 50
              cartheight = 30
              (l,r,t,b) = (-cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2)
              cart = polygon [(l,b), (l,t), (r,t), (r,b)]
              (l',r',t',b') = (-polewidth/2, polewidth/2, polelen - polewidth/2, -polewidth/2)
              pole = color (makeColor 0.8 0.6 0.4 1.0) $ polygon [(l',b'), (l',t'), (r',t'), (r',b')]
              axle = color (makeColor 0.5 0.5 0.8 1.0) $ circleSolid (polewidth / 2)
              poleAssembly = rotate (-(theta st) * 180 / pi) . translate 0 (cartheight / 4) $ pictures [pole, axle]
              cartAssembly = translate (scl * (x st) + screen_width / 2) carty $ pictures [cart, poleAssembly]
              track = color (makeColor 0 0 0 1) $ line [(0,carty), (screen_width, carty)]
              pic = scale (w / screen_width) (h / screen_height) $ pictures [track, cartAssembly]

