#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 06:45:29 PM EDT 2022 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import random

from sklearn.datasets import fetch_20newsgroups

random.seed(42)

#ng = fetch_20newsgroups(categories = ["alt.atheism"])

"""
['alt.atheism',
  'comp.graphics',
  'comp.os.ms-windows.misc',
  'comp.sys.ibm.pc.hardware',
  'comp.sys.mac.hardware',
  'comp.windows.x',
  'misc.forsale',
  'rec.autos',
  'rec.motorcycles',
  'rec.sport.baseball',
  'rec.sport.hockey',
  'sci.crypt',
  'sci.electronics',
  'sci.med',
  'sci.space',
  'soc.religion.christian',
  'talk.politics.guns',
  'talk.politics.mideast',
  'talk.politics.misc',
  'talk.religion.misc']
"""

default = ["She got hiccups and couldn't get rid of them for three hours.","Big Brother is watching.","He snuck into the house after midnight to investigate.","We stayed on a boat last night.","Her mother offered to send her money every month but she was too embarrassed to accept it.","Alone in the big city I began to get homesick.","He was exhausted by the job search but knew he wouldn't get to relax until he had found a job.","It felt great to disconnect with technology.","I like open spaces.","Tragedy brings hope.","The banana was completely brown.","I'm going to the concert.","We've been trying to go there for so long.","Dozens of houses were burned down in that big fire.","She was depressed by the unrelenting forward march of time.","We will make a snowman this winter.","When are we going to start the grill?","Do you eat ice cream?","Go home weirdo!","Keep it up!","There was a big fire in my neighborhood.","He wanted to move in with his girlfriend but she had painted every wall of her house pink and he couldn't stand it.","I like rice and beans for dinner.","There are many things that confuse me about that.","No sport should be called football.","The school principal was so mean that all the children were scared of him.","The spa was called Blue Mercury.","She refused to believe that anyone could actually like mashed potatoes.","She ate a sandwich for lunch which contained turkey cheddar cheese and slices of strawberries.","He slowly began to realize the error of his ways.","She ordered a mass box of fifty-two mac and cheese packets online.","The mailbox was bent and broken and looked like someone had knocked it over on purpose.","Juice was something I never drank.","She was the mother of my children.","Of course I will help you with the move.","That is a pencil.","Where do you come from?","Don't go there!","What are you talking about?","She washed the grapes.","There was a big fire in my neighborhood.","He wanted to move in with his girlfriend but she had painted every wall of her house pink and he couldn't stand it.","I like rice and beans for dinner.","There are many things that confuse me about that.","No sport should be called football.","The school principal was so mean that all the children were scared of him.","The spa was called Blue Mercury.","She refused to believe that anyone could actually like mashed potatoes.","She ate a sandwich for lunch which contained turkey cheddar cheese and slices of strawberries.","He slowly began to realize the error of his ways.","She ordered a mass box of fifty-two mac and cheese packets online.","The mailbox was bent and broken and looked like someone had knocked it over on purpose.","Juice was something I never drank.","She was the mother of my children.","Of course I will help you with the move.","That is a pencil.","Where do you come from?","Don't go there!","What are you talking about?","She washed the grapes.","He is now looking for a bigger house to live in.","Sue was her favorite classmate.","You're pulling my leg.","They employed him as a consultant.","We can go to Paris.","That's a pretty big job to do all by yourself.","Apple picking is not in season.","She wasn't sure if she liked her job or not but she was sure it was better than being unemployed.","She had never read any Nabokov though she knew he was a classic writer.","I was so thirsty that I couldn't wait to get a drink of water.","If you turn right you will see a big building.","I have no idea.","She was shaking like a chihuahua.","He wanted to find out what had happened to his brother.","I am dreaming of a better world.","There is one open baby swing.","A turkey is a little bigger than a chicken.","Sit down and cross your legs please!","What a big supermarket!","Does he go to school?","Soccer is the devil's sport.","The course starts next Sunday.","It is clear that he has made a big mistake.","This recipe has been in my family for 100 years.","I’m the designated driver.","She wasn't sure if she liked him or not.","The dog chased the cat around the block four times.","Meredith made a list of songs she doesn't like.","The gum was stuck under the desk and I couldn't get it off.","She constantly worried about accidentally burning her house down.","I am just extremely proud of him.","She washed the tomatoes.","He said he didn’t want to hotwire a car.","Uncle Joe gave me a red toy truck.","Happy Birthday to my 152 year old lover Canada.","He looked puzzled.","Would you like to have some coffee?","It would be quite impossible to enumerate all the things in existence.","Cows are smelly.","Who's Archibald?"]